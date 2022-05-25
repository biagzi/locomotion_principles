from curses.ascii import SI
import random
import os
import sys
import numpy as np
import subprocess as sub
import pickle

sys.path.insert(0, os.path.abspath('~/locomotion_principles/'))

from base import Sim, ObjectiveDict
from networks import CPPN, DirectGlobalFeature, DirectEncoding
from softbot import Genotype, Phenotype, Population
from tools.algorithms import ParetoOptimizationDiversityChildrenAlternation
from tools.checkpointing import continue_from_checkpoint
from tools.utils import map_genotype_phenotype_CPPN, map_genotype_phenotype_direct_encode

from data_analysis.BasicAnalysisUtils import choose_my_env


sub.call("cp ~/locomotion_principles/_voxcad/voxelyzeMain/voxelyze .", shell=True)
sub.call("chmod 755 voxelyze", shell=True)


def return_ind(EXP_NAME,GEN,SEED,ind_id):
    pickle_loc = "~/locomotion_principles/exp/{0}/{0}_{1}/pickledPops/Gen_{2}.pickle".format(EXP_NAME,SEED,GEN) 

    with open(pickle_loc, 'rb') as handle:
        [optimizer, random_state, numpy_random_state] = pickle.load(handle)

    pop = optimizer.pop
    for ind in pop:
        if ind.id == ind_id:
            return ind
   
    return 

SEED = int(sys.argv[1])
N = int(sys.argv[2])         #4
FROM_ENV = str(sys.argv[3])     #WATER, MARS, EARTH
TO_ENV = str(sys.argv[4])       #WATER, MARS, EARTH
gen = int(sys.argv[5])          
ind = int(sys.argv[6])
DE = sys.argv[7]                #To use CPPN, choose False
GenotDE = sys.argv[8]           #To use CPPN, choose False
MAX_GENS = int(sys.argv[9])     #200
EXTRA_GENS = int(sys.argv[10])  #0
Inovation = str(sys.argv[11])   #True


NUM_RANDOM_INDS = 5
ADD_NEW_IND_FIXED_SHAPE = True

MAX_TIME = 10

#Size of 3d Lattice that can be useds
IND_SIZE = (N,N,N)
MIN_PERCENT_FULL = 0.25

POP_SIZE = 10


INIT_TIME = 2
SIM_TIME = 30 + INIT_TIME  # includes init time
DT_FRAC = 0.9  # 0.3

TIME_TO_TRY_AGAIN = 25
MAX_EVAL_TIME = 61

SAVE_LINEAGES = False
CHECKPOINT_EVERY = 1

EXP_NAME = '{0}_{1}_CPPN'.format(N,FROM_ENV)

try:
    os.mkdir("~/locomotion_principles/TransferenceSelectedShapes/{0}/".format(EXP_NAME))
except OSError:
    print ("Creation of the directory failed. Check if the directory already exist.")

try:
    os.mkdir("/~/locomotion_principles/TransferenceSelectedShapes/{0}/to_{1}".format(EXP_NAME,TO_ENV))
except OSError:
    print ("Creation of the directory failed. Check if the directory already exist.")

try:
    os.mkdir("~/locomotion_principles/TransferenceSelectedShapes/{0}/to_{1}/seed{2}-shape{3}".format(EXP_NAME,TO_ENV,SEED,ind))
except:
    print ("Creation of the directory failed. Check if the directory already exist.")


RUN_NAME = "BestShapeTransf"

SAVE_VXA_EVERY = MAX_GENS + 1
SIMILARITY_THRESHOLD_CHILDREN = None
DIVERSITY_ALTERNATION_PROB = 0 #probability of using the diversity in mutation strategy
MATERIAL_TYPE = 9
MUT_NET_PROB = [0.99999,0,0] #Prob of mutate just phase offset

if GenotDE == 'False':
    class MyGenotype(Genotype):
        def __init__(self):
            Genotype.__init__(self, orig_size_xyz=IND_SIZE)


            self.add_network(CPPN(output_node_names=["phase_offset"]))

            self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>", logging_stats=None)

            self.add_network(CPPN(output_node_names=["shape"]))

            self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=map_genotype_phenotype_CPPN, output_type=int,
                                            dependency_order=["shape"], logging_stats=None)

            self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
                                                            material_if_true="{0}".format(MATERIAL_TYPE), material_if_false="0")
            

            self.add_network(DirectGlobalFeature(output_node_names=["stiffness"],feature_possibilities = [5e4,5e5,5e6,5e7]))
else:
    class MyGenotype(Genotype):
        def __init__(self):
            Genotype.__init__(self, orig_size_xyz=IND_SIZE)

            self.add_network(DirectEncoding(output_node_name="phase_offset", orig_size_xyz=IND_SIZE, symmetric=False,
                                        upper_bound=0.5, lower_bound=-0.5))
            self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>", logging_stats=None)


            self.add_network(DirectEncoding(output_node_name="material", orig_size_xyz=IND_SIZE, symmetric=False,
                                        upper_bound=1, lower_bound=-1))
            self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=map_genotype_phenotype_direct_encode, output_type=int,
                                logging_stats=None, dependency_order="material")
            self.to_phenotype_mapping.add_output_dependency(name="material", dependency_name=None, requirement=None,
                                                            material_if_true="{0}".format(MATERIAL_TYPE), material_if_false="0")
            

            self.add_network(DirectGlobalFeature(output_node_names=["stiffness"],feature_possibilities = [5e4,5e5,5e6,5e7]))

        def express(self,bias_param = 1):
            """Calculate the genome networks outputs, the physical properties of each voxel for simulation"""
            #R: This looping gives to the phenotype mapping the values calculeted in the genotype

            for network in self:
                for name in network.output_node_names: 
                    if name in self.to_phenotype_mapping:  #R: if there is a mapping between them (just in the case of phase offset)
                        if network.direct_encoding: 
                            self.to_phenotype_mapping[name]["state"] = network.values 
                        elif network.direct_global_feature:
                            self.to_phenotype_mapping[name]["state"] = network.feature

            #R: Enter inside this loop just if is 'material' (will create the material matrix):        
            for name, details in self.to_phenotype_mapping.items():
                if details["dependency_order"] is not None: #R: this is the case of 'shape'
                    details["state"] = details["func"](self) #R: material[state] = func:make_material_tree
            #R: in the end, the ['material']['state'] receives the result of make_material_tree(self) func


class MyPhenotype(Phenotype):

    #Check if a randomly generated phenotype is valid
    def is_valid(self, min_percent_full=MIN_PERCENT_FULL):

        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any(): 
                return False
            if name == "material": 
                state = details["state"]
                num_vox = np.sum(state > 0)
                if num_vox < np.product(self.genotype.orig_size_xyz) * min_percent_full:
                    return False
                if np.sum(state == MATERIAL_TYPE) == 0:  # make sure has at least one muscle voxel for movement
                    return False
        return True

fixed_shape_model = return_ind(EXP_NAME,gen,SEED,ind)
ORIGINAL_STIFF = fixed_shape_model.genotype[2].feature

if TO_ENV == 'WATER':
    STIFF = 5e5
elif TO_ENV == 'EARTH' or TO_ENV == 'MARS':
    STIFF = 5e7

fixed_shape_model.genotype[2].feature = STIFF
fixed_shape_model.dominated_by = []  # other individuals in the population that are superior according to evaluation
fixed_shape_model.pareto_level = 0
fixed_shape_model.selected = 0  # survived selection
fixed_shape_model.variation_type = "copied from best of"  # (from parent)


RUN_DIR = "~/locomotion_principles/TransferenceSelectedShapes/{0}/to_{1}/seed{2}-shape{3}/CPPN".format(EXP_NAME,TO_ENV,SEED,ind)

if not os.path.isfile(RUN_DIR + "/pickledPops/Gen_0.pickle"):

    random.seed(SEED)
    np.random.seed(SEED)

    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

    my_env = choose_my_env(TO_ENV)

    my_objective_dict = ObjectiveDict()

    #fitness is always the first objective
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<normAbsoluteDisplacement>",logging_only=False
                                    )

    my_objective_dict.add_objective(name="age", maximize=False, tag=None, logging_only=False)

    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POP_SIZE)

    for i in range(len(my_pop.individuals)):
        my_pop.individuals[i].genotype = fixed_shape_model.genotype
        my_pop.individuals[i].parent_fitness = fixed_shape_model.fitness
        my_pop.individuals[i].parent_age = fixed_shape_model.age
        my_pop.individuals[i].genotype.express()


    my_optimization = ParetoOptimizationDiversityChildrenAlternation(my_sim, my_env, my_pop,
                    SIMILARITY_THRESHOLD_CHILDREN,DIVERSITY_ALTERNATION_PROB,MUT_NET_PROB)
    
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_VXA_EVERY,
                        add_new_ind_fixed_shape = ADD_NEW_IND_FIXED_SHAPE,
                        fixed_shape = fixed_shape_model.genotype.networks[1],TO_ENV = TO_ENV)

else:
    continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                             max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                             checkpoint_every=CHECKPOINT_EVERY,save_vxa_every=SAVE_VXA_EVERY,
                             add_new_ind_fixed_shape = ADD_NEW_IND_FIXED_SHAPE,
                                fixed_shape = fixed_shape_model.genotype.networks[1],TO_ENV = TO_ENV)
