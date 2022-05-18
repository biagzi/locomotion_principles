from curses.ascii import SI
import random
import os
import sys
import numpy as np
import subprocess as sub
from functools import partial

from base import Sim, Env, ObjectiveDict
from networks import CPPN, DirectGlobalFeature
from softbot import Genotype, Phenotype, Population
from tools.algorithms import ParetoOptimizationDiversityChildrenAlternation
from tools.checkpointing import continue_from_checkpoint
from tools.utils import map_genotype_phenotype_CPPN


sub.call("cp ~/reconfigurable_organisms/_voxcad/voxelyzeMain/voxelyze .", shell=True)
sub.call("chmod 755 voxelyze", shell=True)

SEED = int(sys.argv[1])
MAX_TIME = float(sys.argv[2]) #max_hours_runtime


######################### Experiment Gravity Environment and Size #########################
###########################################################################################
N = 4 #6,8  -> Size of 3d Lattice that can be used
IND_SIZE = (N,N,N)  


GRAV_ACC = -0.1 #-3.721, -9.81
RUN_NAME = "Aquatic" #"Martians", "Terrestrial"
RUN_DIR = "~/{0}_Water_CPPN/{0}_Water_CPPN_{1}".format(N,SEED) #Mars, Earth


#########################################EA PARAMETERS ##################################################

MIN_PERCENT_FULL = 0.25 #of the N^3 lattice
POP_SIZE = 50
MAX_GENS = 1500
NUM_RANDOM_INDS = 1

INIT_TIME = 2
SIM_TIME = 30 + INIT_TIME  # includes init time
TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
FREQ = 2

DT_FRAC = 0.9  

VOXEL_SIZE = 0.01  # meters

TIME_TO_TRY_AGAIN = 25
MAX_EVAL_TIME = 61

SAVE_VXA_EVERY = MAX_GENS + 1
SAVE_LINEAGES = True
CHECKPOINT_EVERY = 1
EXTRA_GENS = 0

TIME_BETWEEN_TRACES = 0.0 

SIMILARITY_THRESHOLD_CHILDREN = 0.95
DIVERSITY_ALTERNATION_PROB = 0.5
MATERIAL_TYPE = 9
MUT_NET_PROB = [1/2,1/2,1/5]

############################################################################################################
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



if not os.path.isfile(RUN_DIR + "/pickledPops/Gen_0.pickle"):

    random.seed(SEED)
    np.random.seed(SEED)

    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

    my_env = Env(temp_amp=TEMP_AMP, frequency=FREQ,
                 lattice_dimension=VOXEL_SIZE, grav_acc=GRAV_ACC,
                 )

    my_objective_dict = ObjectiveDict()

    #fitness is always the first objective
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<normAbsoluteDisplacement>",logging_only=False
                                    )

    my_objective_dict.add_objective(name="age", maximize=False, tag=None, logging_only=False)

    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POP_SIZE)

    my_optimization = ParetoOptimizationDiversityChildrenAlternation(my_sim, my_env, my_pop,
                    SIMILARITY_THRESHOLD_CHILDREN,DIVERSITY_ALTERNATION_PROB,MUT_NET_PROB)
    
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)

else:
    continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                             max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                             checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_VXA_EVERY,
                             save_lineages=SAVE_LINEAGES)
