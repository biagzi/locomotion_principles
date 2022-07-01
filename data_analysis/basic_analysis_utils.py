import pickle
import cPickle
import sys
import os
from mpl_toolkits.mplot3d import Axes3D


sys.path.insert(0, os.path.abspath('~/locomotion_principles'))

from softbot import Genotype, Phenotype
from base import Env


MyPhenotype = Phenotype
MyGenotype = Genotype

# return the pop file using the pickle file
def open_pop_from_pickle(exp_name,GEN,returnAllOptimizer = False):

    PICKLE_DIR = "~/locomotion_principles/exp/{0}/pickledPops".format(exp_name)

    pickle = "{0}/Gen_{1}.pickle".format(PICKLE_DIR, GEN)

    handle = open(pickle, 'rb') #rb = read and binary
    [optimizer, random_state, numpy_random_state] = cPickle.load(handle)

    pop = optimizer.pop

    if returnAllOptimizer == False:
        return pop

    else: 
        return optimizer

def generate_dicts_allrobots_per_seed_with_stiff(EXP_NAME,MAX_GEN,SEED_INIT,SEED_END,encode = 'ASCII'):

    """Go in all seeds of an experiment and generates a dict with all inds and this infos: [id]: {fitness, shape, phase_offset, parent_id, stiff} 
    for each generation. It simplifies the access to this data for further analysis.
    
    """

    for seed in range(SEED_INIT,SEED_END+1):
        exp_name = '{0}/{0}_{1}'.format(EXP_NAME,seed)
        
        if os.path.isfile("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed)) is False:

            all_gen_dict = {}
            list_of_ids_this_seed = []

            for gen in range(0,MAX_GEN):
                each_gen_dict = {}

                try: 
                    pop = open_pop_from_pickle(exp_name,gen)
                    
                    for ind in pop:
                        if ind.id not in list_of_ids_this_seed: #to avoid add the same ind twice.
                            try:
                                this_robot_stiffness = ind.genotype[2].feature
                            except:
                                this_robot_stiffness = 0
                            each_gen_dict[ind.id] = [ind.fitness, ind.genotype.to_phenotype_mapping['material']['state'], ind.genotype.to_phenotype_mapping['phase_offset']['state'],ind.parent_id,this_robot_stiffness]
                            list_of_ids_this_seed.append(ind.id)


                    all_gen_dict[gen] = each_gen_dict


                    if gen%100 == 0:
                        with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed), 'wb') as handle:
                            pickle.dump(all_gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print gen
                
                except:
                    print 'gen {0} seed does not work'.format(gen,seed)
                    return 
                
        else:
            with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed), 'rb') as handle:
                if encode == "ASCII":
                    all_gen_dict = pickle.load(handle)
                else:
                    all_gen_dict = pickle.load(handle,encoding=encode)

            if MAX_GEN - 1 in all_gen_dict.keys():
                print 'seed {0} is updated'.format(seed)

            else:
                print "starting update in gen {0} to be finished in MAXGEN:{1} -1".format(max(all_gen_dict.keys()),MAX_GEN)
                list_of_ids_this_seed = []
                
                for gen in all_gen_dict.keys():
                    list_of_ids_this_seed.extend(all_gen_dict[gen].keys())

                each_gen_dict = all_gen_dict[max(all_gen_dict.keys())]

                for gen in range(max(all_gen_dict.keys()),MAX_GEN):
                    #list_of_ids = each_gen_dict.keys()
                    each_gen_dict = {}

                    try:
                        pop = open_pop_from_pickle(exp_name,gen)

                        for ind in pop:
                            if ind.id not in list_of_ids_this_seed: #to avoid add the same ind twice.
                                try:
                                    this_robot_stiffness = ind.genotype[2].feature
                                except:
                                    this_robot_stiffness = 0
                                each_gen_dict[ind.id] = [ind.fitness, ind.genotype.to_phenotype_mapping['material']['state'], ind.genotype.to_phenotype_mapping['phase_offset']['state'],ind.parent_id,this_robot_stiffness]
                                list_of_ids_this_seed.append(ind.id)


                        all_gen_dict[gen] = each_gen_dict


                        if gen%100 == 0:
                            with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed), 'wb') as handle:
                                pickle.dump(all_gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            print gen
                    
                    except:
                        print 'gen {0} seed {1} does not work'.format(gen,seed)
                        return 



        with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed), 'wb') as handle:
            pickle.dump(all_gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print 'finished seed {0}'.format(seed)

def choose_my_env(new_env):
    if new_env == 'WATER':
        TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
        FREQ = 2
        GRAV_ACC = -0.1
        VOXEL_SIZE = 0.01  # meters

    elif new_env == 'EARTH':
        TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
        FREQ = 2
        GRAV_ACC = -9.81
        VOXEL_SIZE = 0.01  # meters

    elif new_env == 'MARS':
        TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
        FREQ = 2
        GRAV_ACC = -3.721
        VOXEL_SIZE = 0.01  # meters
        
    my_env = Env(temp_amp=TEMP_AMP, frequency=FREQ,lattice_dimension=VOXEL_SIZE, grav_acc=GRAV_ACC)
    return my_env


def return_shape_po_fitness(seed, EXP_NAME,MAX_GEN,return_stiff = False, encode = "ASCII",exact = False):
    
    """
    Returns all_fit, all_stiff, all_nclusters, all_Nclusters_voxels (Nclusters/Nvoxels)

    """

    all_fits_shape_po = {}

    if os.path.isfile("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed)) is True:
        with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed), 'rb') as handle:
            if encode == "ASCII":
                all_gen_dict = pickle.load(handle)
            else:
                all_gen_dict = pickle.load(handle,encoding=encode)
    else:
        print ('do not found seed')
        print ("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(EXP_NAME,seed))
        return
    
    if return_stiff == False:
        if exact == False:
            for gen in all_gen_dict:
                if gen <= MAX_GEN:
                    for ind in all_gen_dict[gen]:
                        fit, shape, po = all_gen_dict[gen][ind][0], all_gen_dict[gen][ind][1], all_gen_dict[gen][ind][2]
                        all_fits_shape_po[ind] = {'fit':fit,'shape':shape,'po':po}
        else:
            for gen in all_gen_dict:
                if gen == MAX_GEN:
                    for ind in all_gen_dict[gen]:
                        fit, shape, po = all_gen_dict[gen][ind][0], all_gen_dict[gen][ind][1], all_gen_dict[gen][ind][2]
                        all_fits_shape_po[ind] = {'fit':fit,'shape':shape,'po':po}
    else:
        if exact == False:
            for gen in all_gen_dict:
                if gen <= MAX_GEN:
                    for ind in all_gen_dict[gen]:
                        fit, shape, po, stiff = all_gen_dict[gen][ind][0], all_gen_dict[gen][ind][1], all_gen_dict[gen][ind][2],all_gen_dict[gen][ind][4]
                        all_fits_shape_po[ind] = {'fit':fit,'shape':shape,'po':po,'stiff':stiff}
        else:
            for gen in all_gen_dict:
                if gen == MAX_GEN:
                    for ind in all_gen_dict[gen]:
                        fit, shape, po, stiff = all_gen_dict[gen][ind][0], all_gen_dict[gen][ind][1], all_gen_dict[gen][ind][2],all_gen_dict[gen][ind][4]
                        all_fits_shape_po[ind] = {'fit':fit,'shape':shape,'po':po,'stiff':stiff}
    return all_fits_shape_po



def return_fit_stiff_nclusters(SEED, TITLE_NAME,MAX_GEN,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,encode = "ASCII"):
    """
    Returns all_fits_shape_po[ind] = {'fit','shape','po'}

    if exact = True: all_fits_shape_po[ind] =  {'fit','shape','po','stiff'}
    """

    from save_open_clustering import open_cluster_seed

    all_fit = []
    all_stiff = []
    all_nclusters = []
    all_Nclusters_voxels = []

    if os.path.isfile("/home/{2}/reconfigurable_organisms/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(TITLE_NAME,SEED,COMPUTER_NAME)) is True:
        with open("/home/{2}/reconfigurable_organisms/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(TITLE_NAME,SEED,COMPUTER_NAME), 'rb') as handle:
            if encode == "ASCII":
                all_gen_dict = pickle.load(handle)
            else:
                all_gen_dict = pickle.load(handle,encoding=encode)
    else:
        print ('do not found seed')
        print ("/home/{2}/reconfigurable_organisms/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(TITLE_NAME,SEED,COMPUTER_NAME))
        return
    
    clustering_allX_results = open_cluster_seed(SEED,TITLE_NAME,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,encode)

    for gen in all_gen_dict:
        if gen <= MAX_GEN:
            for ind in all_gen_dict[gen]:
                all_fit.append(all_gen_dict[gen][ind][0])
                all_stiff.append(all_gen_dict[gen][ind][4])
                all_nclusters.append(clustering_allX_results[ind]['n_clusters'])
                all_Nclusters_voxels.append(clustering_allX_results[ind]['n_clusters']/np.sum(np.sum(np.sum(all_gen_dict[gen][ind][1] > 0))))

    return all_fit, all_stiff, all_nclusters,all_Nclusters_voxels