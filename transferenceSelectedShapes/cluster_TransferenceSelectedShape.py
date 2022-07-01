import os
import sys
import numpy as np
import subprocess as sub
import time
import pandas as pd
import seaborn as sns
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
import pickle
from natsort import natsorted


PC_NAME = 'renata'
sys.path.insert(0, os.path.abspath('/home/{0}/locomotion_principles'.format(PC_NAME)))
from softbot import Genotype, Phenotype
MyPhenotype = Phenotype
MyGenotype = Genotype

PICKLE_LOCATION = '/scratch/renata'

def ClusteringAllEPSandEPSPO(ind_seed_shape,ind_exp_name,ind_idInOptimization,EPS_LIST,EPS_PO_LIST,EXP_NAME,TO_ENV,SIZE,min_samples,COARSE_GRAIN,SELECT_ROBOT=False,RUN_DIR=None):
    """
    Do the clustering of each EPS and EPS_PO (if it has not been done yet), 
    run the simulation of the caricature to check the results and check if it is good enough
    If it is, abort clustering search for params
    If not, continue to do 
    Do for each robot and save results in
    ClustersToCaricature_EXP_NAME.pickle
    """
    sys.path.insert(0, os.path.abspath('/home/renata/locomotion_principles/data_analysis/Clustering/')) 
    from clustering_algorithm import core_clustering_algorithm

    save_caricatures_dir = "/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/Caricatures_{1}/{2}".format(PC_NAME,SELECT_ROBOT,EXP_NAME)

    if os.path.isfile('{0}/ClustersToCaricature_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT)) is True:
        with open('{0}/ClustersToCaricature_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT), 'rb') as handle:
            all_clusters_results = pickle.load(handle)
    else:
        all_clusters_results = {}

    if os.path.isfile('{0}/CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT)) is True:
        with open('{0}/CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT), 'rb') as handle:
            robots_caricatures_info = pickle.load(handle)
    else:
        robots_caricatures_info = {}

    RETURN_IND = True
    FOUND = False
    for eps in EPS_LIST:
        if eps not in list(all_clusters_results.keys()):
            all_clusters_results[eps] = {}
        for eps_po in EPS_PO_LIST:
            if eps_po not in list(all_clusters_results[eps]):
                all_clusters_results[eps][eps_po] = {}
            if ind_seed_shape not in list(all_clusters_results[eps][eps_po]):
                if RETURN_IND:
                    WHAT = 'clusterANDsimulation'
                    factor = 10
                    optimizer,ind, X_original, X_po = ind_to_cluster_and_simulate(SIZE,ind_idInOptimization, ind_exp_name, factor,WHAT,SELECT_ROBOT=SELECT_ROBOT)
                    RETURN_IND = False

                cluster_results = all_clusters_results[eps][eps_po]
                cluster_results = core_clustering_algorithm(X_original,X_po,eps,eps_po,min_samples,COARSE_GRAIN,cluster_results,ind_seed_shape,factor = 10,
                                            save_gen = [False,None],
                                            save_shape = [False,None],
                                            save_po = [False,None],
                                            save_X_original= False
                                            )
                all_clusters_results[eps][eps_po] = cluster_results
            
                with open('{0}/ClustersToCaricature_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT), 'wb') as handle:
                    pickle.dump(all_clusters_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Finished cluster {0}-{1}'.format(eps,eps_po))
            else:
                if RETURN_IND:
                    optimizer,ind, X_original = None,None,None
            this_robot_caricature,DO_CHECK,lenX_Original = CaricaturesSimulation(robots_caricatures_info,optimizer,ind, X_original,all_clusters_results,ind_idInOptimization,eps,eps_po,ind_seed_shape,SIZE,ind_exp_name,EXP_NAME,TO_ENV,SELECT_ROBOT=SELECT_ROBOT,RUN_DIR=RUN_DIR)
            print('Finished caricature {0}-{1}'.format(eps,eps_po))
            if DO_CHECK:
                FOUND = FOUND_CARICATURE_GOOD_ENOUGH(this_robot_caricature,lenX_Original)
                if FOUND:
                    break
        if FOUND:
            break


def CaricaturesSimulation(robots_caricatures_info,optimizer,ind, X_original,all_clusters_results,ind_idInOptimization,eps, eps_po,ind_seed_shape,SIZE,ind_exp_name, EXP_NAME,TO_ENV,SELECT_ROBOT=False,RUN_DIR=None):
    """
    - Based on the robot eps and eps_po of ClustersToCaricature_EXP_NAME_toENV.pickle, 
    
    - Write a caricature and simulate it, saving the fitness of each one and n_clusters 
    in CaricaturesResults_TITLENAME.pickle

    """
    
    sys.path.insert(0, os.path.abspath('/home/{0}/locomotion_principles/data_analysis/Clustering'.format(PC_NAME)))
    from generate_compare_caricatures import list_to_matrix_po, replace_po_by_mean_po, mean_po_per_cluster_index, write_voxelyze_file_caricature

    sys.path.insert(0, os.path.abspath('/home/{0}/locomotion_principles'.format(PC_NAME)))
    from tools.read_write_voxelyze import write_voxelyze_file

    save_caricatures_dir = "/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/Caricatures_{1}/{2}".format(PC_NAME,SELECT_ROBOT,EXP_NAME)

    try:
        this_robot_caricature = robots_caricatures_info[ind_seed_shape]
    except:
        this_robot_caricature = {}
    
    lenXOriginal = None
    DO_CHECK = False
    #Do Original Fit
    if 'OriginalRobot' not in list(this_robot_caricature.keys()):
        DO_CHECK = True
        if optimizer == None:
            WHAT = 'simulation'
            factor = 10
            optimizer,ind, X_original = ind_to_cluster_and_simulate(SIZE,ind_idInOptimization, ind_exp_name, factor,WHAT,SELECT_ROBOT=SELECT_ROBOT)
        
        lenXOriginal = len(X_original)
        
        id_corrected = '{0}'.format(ind.id).zfill(5)
        save_name = "{0}_Original_{1}".format(TO_ENV,ind_seed_shape)
        write_voxelyze_file(optimizer.sim,optimizer.env[0], ind,
                                            save_caricatures_dir, save_name)
        robot_vxa = '{0}/voxelyzeFiles/{1}--id_{2}.vxa'.format(save_caricatures_dir,save_name,id_corrected)
        robot_out = '{0}/fitnessFiles/softbotsOutput--id_{1}.xml'.format(save_caricatures_dir,id_corrected)
        
        if os.path.isfile(robot_out) is True: #erase older files if they exist
            sub.call("rm " + robot_out, shell=True)
        sub.Popen("/home/{0}/locomotion_principles/_voxcad/voxelyzeMain/voxelyze -f ".format(PC_NAME) + robot_vxa, shell=True)
        
        time.sleep(3)
        count = 0
        while os.path.isfile(robot_out) is False:
            time.sleep(2)
            count += 1
            if count%10:
                print(count)
            if count > 1000:
                print('Problem in simulation of {0}'.format(robot_vxa))
                exit()

        tag = '<normAbsoluteDisplacement>'
        this_robot = open(robot_out)
        for line in this_robot:
            if tag in line:
                this_robot_caricature['OriginalRobot'] = {'Fit':float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])}
        sub.call("rm " + robot_out, shell=True)    
        sub.call("rm " + robot_vxa, shell=True)
        with open('{0}/CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT), 'wb') as handle:
            pickle.dump(robots_caricatures_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #Do caricatures
    if ind_seed_shape in list(all_clusters_results[eps][eps_po]):
        tag_eps_epspo = '{0}-{1}'.format(eps,eps_po)
        if tag_eps_epspo not in list(this_robot_caricature.keys()):
            print('Start caricature {0}-{1}'.format(eps,eps_po))
            DO_CHECK = True
            if optimizer == None:
                WHAT = 'simulation'
                factor = 10
                optimizer,ind, X_original = ind_to_cluster_and_simulate(SIZE,ind_idInOptimization, ind_exp_name, factor,WHAT,SELECT_ROBOT=SELECT_ROBOT)
            lenXOriginal = len(X_original)
            data = deepcopy(X_original)
            labels = all_clusters_results[eps][eps_po][ind_seed_shape]['labels']
            clusters_mean = mean_po_per_cluster_index(data ,labels,3)
            X = replace_po_by_mean_po(clusters_mean,data,labels,0.1)
            po_matrix = np.array(list_to_matrix_po(labels,X,[SIZE,SIZE,SIZE]))
            itemsdeepcopy = deepcopy(ind.genotype.to_phenotype_mapping.items())
            itemsdeepcopy[1][1]['state'] = po_matrix

            id_corrected = '{0}'.format(ind.id).zfill(5)
            save_name = "{0}_eps{1}epspo{2}_{3}".format(TO_ENV,eps,eps_po,ind_seed_shape)
            
            #Create vxa file to simulate
            write_voxelyze_file_caricature(optimizer.sim,optimizer.env[0], ind,
                                            save_caricatures_dir, save_name, itemsdeepcopy)
            
            robot_vxa = '{0}/voxelyzeFiles/{1}--id_{2}.vxa'.format(save_caricatures_dir,save_name,id_corrected)
            robot_out = '{0}/fitnessFiles/softbotsOutput--id_{1}.xml'.format(save_caricatures_dir,id_corrected)

            #Simulate
            if os.path.isfile(robot_out) is True: #erase older files if they exist
                sub.call("rm " + robot_out, shell=True)
            
            sub.Popen("/home/{0}/locomotion_principles/_voxcad/voxelyzeMain/voxelyze -f ".format(PC_NAME) + robot_vxa, shell=True)
            time.sleep(2)
            count = 0
            while os.path.isfile(robot_out) is False:
                time.sleep(2)
                count += 1
                if count%10:
                    print(count)
                if count > 1000:
                    print('Problem in simulation of {0}'.format(robot_vxa))
                    exit()

            tag = '<normAbsoluteDisplacement>'
            this_robot = open(robot_out)
            for line in this_robot:
                if tag in line:
                    this_robot_caricature[tag_eps_epspo] = {'Fit':float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])}
                    
            this_robot_caricature[tag_eps_epspo]['n_clusters'] = all_clusters_results[eps][eps_po][ind_seed_shape]['n_clusters']

            sub.call("rm " + robot_out, shell=True)    
            sub.call("rm " + robot_vxa, shell=True)

            robots_caricatures_info[ind_seed_shape] = this_robot_caricature
            time.sleep(3)
            with open('{0}/CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT), 'wb') as handle:
                pickle.dump(robots_caricatures_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    

    return this_robot_caricature, DO_CHECK, lenXOriginal


def SelectBestCaricature(ind_seed_shape,EXP_NAME,TO_ENV,Nvoxels,RUN_DIR,SELECT_ROBOT):
    
    save_caricatures_dir = "/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/Caricatures_{1}/{2}".format(PC_NAME,SELECT_ROBOT,EXP_NAME)


    if os.path.isfile('{0}/CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT)) is True:
        with open('{0}/CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT), 'rb') as handle:
            robots_caricatures_info = pickle.load(handle)
    else:
        print('CaricaturesResults_{1}_to{2}_{3}-{4}.pickle'.format(save_caricatures_dir,EXP_NAME,TO_ENV,RUN_DIR,SELECT_ROBOT))
        return
    
    this_robot_caricature = robots_caricatures_info[ind_seed_shape]

    tags = []
    fitness = []
    nclusters = []
    fitnessdiff = []

    OriginalFit = this_robot_caricature['OriginalRobot']['Fit']

    for key in this_robot_caricature:
        if key != 'OriginalRobot':

            tags.append(key)
            fitness.append(this_robot_caricature[key]['Fit'])
            nclusters.append(this_robot_caricature[key]['n_clusters'])
            fitnessdiff.append(np.abs(OriginalFit - this_robot_caricature[key]['Fit']))

    results = pd.DataFrame({'Fitness':fitness,'NClusters':nclusters,'Eps-EpsPo':tags,'FitDiff':fitnessdiff})
    results = results.sort_values(by = ['FitDiff'],ascending=True).reset_index(drop=True)

    for i in range(len(results['NClusters'])): #The winner is the robot with the fitness closest to the Original and Ncluster != Nvoxels
        if results['NClusters'][i] < Nvoxels:
            winner_nclus = results['NClusters'][i]
            winner_fit = results['Fitness'][i]
            best_caricature_params = results['Eps-EpsPo'][i]
            CaricatureFitness = results['Fitness'][i]
            break

    for i in range(len(results['NClusters'])): #Check if there is a robot with very similar fitness to the winner and a smaller Ncluster

        dif_fit = np.abs(100*(winner_fit - results['Fitness'][i])/winner_fit)
        nclus = results['NClusters'][i]

        if dif_fit <= 10 and nclus < winner_nclus: 
            best_caricature_params = results['Eps-EpsPo'][i]
            winner_nclus = nclus
            CaricatureFitness = results['Fitness'][i]
  

    return best_caricature_params, CaricatureFitness
    

def FOUND_CARICATURE_GOOD_ENOUGH(this_robot_caricature,NVoxels):
    
    OriginalFit = this_robot_caricature['OriginalRobot']['Fit']

    FOUND_GOOD_CARICATURE = False
    for key in this_robot_caricature:
        if key != 'OriginalRobot':
            dif_fit = np.abs(this_robot_caricature[key]['Fit'] - OriginalFit)
            n_clusters = this_robot_caricature[key]['n_clusters']
            if dif_fit < 5 and n_clusters <= int(0.15*NVoxels):
                FOUND_GOOD_CARICATURE = True
                break
                

    return FOUND_GOOD_CARICATURE

def ClusterSimulateAndSelectBestCaricature(ind_seed_shape,ind_exp_name,ind_idInOptimization,SIZE,EPS_LIST, EPS_PO_LIST,min_samples,COARSE_GRAIN, cluster_results,EXP_NAME,TO_ENV,SELECT_ROBOT=False,DoClusterAndSim=True,RUN_DIR=None):
    """
    Calls
    - Cluster core algorithm + 
    - Simulation of Caricature + 
    - Selection of the best one 
    and
    returns as the cluster_results
    """

    sys.path.insert(0, os.path.abspath('/home/renata/locomotion_principles/data_analysis/Clustering')) 
    from save_open_clustering import  core_clustering_algorithm
    

    if DoClusterAndSim:
        ClusteringAllEPSandEPSPO(ind_seed_shape,ind_exp_name,ind_idInOptimization,EPS_LIST,EPS_PO_LIST,EXP_NAME,TO_ENV,SIZE,min_samples,COARSE_GRAIN,SELECT_ROBOT=SELECT_ROBOT,RUN_DIR=RUN_DIR)
    
    time.sleep(1)
    WHAT = 'cluster+infos'
    factor = 10
    X_original, X_po, shape, po = ind_to_cluster_and_simulate(SIZE,ind_idInOptimization, ind_exp_name, factor ,WHAT,SELECT_ROBOT=SELECT_ROBOT)

    best_caricature_params, CaricatureFitness = SelectBestCaricature(ind_seed_shape,EXP_NAME,TO_ENV,len(X_original),RUN_DIR,SELECT_ROBOT)
    eps = float(best_caricature_params[:best_caricature_params.find('-')])
    eps_po = float(best_caricature_params[best_caricature_params.find('-')+1:])


    cluster_results = core_clustering_algorithm(X_original,X_po,eps,eps_po,min_samples,COARSE_GRAIN,cluster_results,ind_seed_shape,factor = 10,
                                        save_gen = [False,None],
                                        save_shape = [True,shape],
                                        save_po = [True,po],
                                        save_X_original= True,
                                        saveEpsEpsPo = True,
                                        saveCaricatureFitness = [True,CaricatureFitness]
                                        )
    
    return cluster_results

def ClusterizationSimulationSelection_MainAlgorithm(FROM_ENV,TO_ENV,SIZE,EPS_LIST, EPS_PO_LIST,min_samples,COARSE_GRAIN,DoClusterAndSim=True,SELECT_ROBOT=False,RUN_DIR = None):
    """
    Main algorithm that proceeds the Clustering + Simulation of Caricatures + Selection of the Best Caricature
    for each Transfered Robots for each robot from each env.
    It calls the function ClusterSimulateAndSelectBestCaricature FOR EACH FROM_ENV AND TO_ENV robot
    and saves the results.

    """

    if RUN_DIR == 'CPPN' or RUN_DIR == 'CPPN_Inovation':
        EXP_NAME = 'Final_{0}_{1}'.format(SIZE,FROM_ENV)
    else:
        EXP_NAME = 'Final_{0}_{1}_DirectEncode'.format(SIZE,FROM_ENV)
    print('Entered in {0}'.format(EXP_NAME))

        
    ouput_files_path = "/home/{0}/locomotion_principles/data_analysis/CMVoxelTrajectory".format(PC_NAME)
    OUTPUT_IN_RECONFIG = "{0}/OuputFiles/{1}-{2}/{3}".format(ouput_files_path,SELECT_ROBOT,RUN_DIR,EXP_NAME)

    count = 0      
    for output_file in glob("{0}/*.xml".format(OUTPUT_IN_RECONFIG)):
        seed = output_file[output_file.find('seed')+4:output_file.find('_id')]
        ind_id = output_file[output_file.find('_id')+3:output_file.find('_toenv')]
        ind_idInOptimization = int(output_file[output_file.find('id_')+3:output_file.find('.xml')])
        ind_seed_shape = 'seed{0}-shape{1}'.format(seed,ind_id)
        to_env = output_file[output_file.find('toenv')+5:output_file.find('--')]
        ind_exp_name = "TransferenceSelectedShapes/{0}/to_{1}/{2}/{3}/".format(EXP_NAME,TO_ENV,ind_seed_shape,RUN_DIR)
        if TO_ENV == to_env:
            count += 1
            DO = True
            print('Start Ind {0}'.format(ind_seed_shape))

            if os.path.isfile('/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/ClusterOptWithCaricaturesCheck{1}_{2}-{3}.pickle'.format(PC_NAME,EXP_NAME,RUN_DIR,SELECT_ROBOT)) is True:
                with open('/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/ClusterOptWithCaricaturesCheck{1}_{2}-{3}.pickle'.format(PC_NAME,EXP_NAME,RUN_DIR,SELECT_ROBOT), 'rb') as handle:
                    TransfSelectedShapeClustering = pickle.load(handle)
            else:
                TransfSelectedShapeClustering = {}
            if ind_seed_shape not in TransfSelectedShapeClustering.keys():
                    TransfSelectedShapeClustering[ind_seed_shape] = {} 
            else:
                if '{0}'.format(TO_ENV) in TransfSelectedShapeClustering[ind_seed_shape].keys():
                    if "CaricatureFitness" in TransfSelectedShapeClustering[ind_seed_shape]['{0}'.format(TO_ENV)].keys():
                        cluster_results = TransfSelectedShapeClustering[ind_seed_shape]['{0}'.format(TO_ENV)]
                        if cluster_results['n_clusters'] <= int(0.1*len(cluster_results['X_original'])):
                            DO = False
            
            if DO:
                cluster_results = {}
                cluster_results = ClusterSimulateAndSelectBestCaricature(ind_seed_shape,ind_exp_name,ind_idInOptimization,SIZE,EPS_LIST, EPS_PO_LIST,min_samples,COARSE_GRAIN, cluster_results,EXP_NAME,TO_ENV,
                                                DoClusterAndSim=DoClusterAndSim,SELECT_ROBOT=SELECT_ROBOT,RUN_DIR=RUN_DIR)
                TransfSelectedShapeClustering[ind_seed_shape]['{0}'.format(TO_ENV)] = cluster_results[ind_seed_shape]

                

                with open('/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/ClusterOptWithCaricaturesCheck{1}_{2}-{3}.pickle'.format(PC_NAME,EXP_NAME,RUN_DIR,SELECT_ROBOT), 'wb') as handle:
                    pickle.dump(TransfSelectedShapeClustering, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Finished {0}'.format(EXP_NAME))
            print('Finished {0}/50 - {2} to_env{1}'.format(count,TO_ENV,EXP_NAME))


def ind_to_cluster_and_simulate(SIZE,ind_idInOptimization, ind_exp_name, factor,WHAT,SELECT_ROBOT = False):
    """
    Given the ind_exp_name, open pickle and returns what is needed for clusterization or simulation
    """

    sys.path.insert(0, os.path.abspath('/home/renata/locomotion_principles/data_analysis'))
    from lineages_and_pt import open_pop_from_pickle



    if SELECT_ROBOT=='BEST':
        pop = open_pop_from_pickle(ind_exp_name,199,PICKLE_LOCATION)
        for inds_opti in pop:
            if inds_opti.id == ind_idInOptimization:
                ind = inds_opti

    elif SELECT_ROBOT=='WORST':
        if ind_idInOptimization <= 9:
            gen = 0
        else:
            gen = int((ind_idInOptimization-9)/15.) + 1
            if int((ind_idInOptimization-9)/15.) == (ind_idInOptimization-9)/15.:
                gen = gen -1
                
        pop = open_pop_from_pickle(ind_exp_name,gen,PICKLE_LOCATION)
        for inds_opti in pop:
            if inds_opti.id == ind_idInOptimization:
                ind = inds_opti

    try:
        shape = ind.genotype.to_phenotype_mapping['material']['state']
    except:
        print(ind_idInOptimization,(ind_idInOptimization-9)/15.,gen)
    po = ind.genotype.to_phenotype_mapping['phase_offset']['state']

    X_original = []
    X_po = []
    for x in range(SIZE):
        for y in range(SIZE):
            for z in range(SIZE):
                if shape[x][y][z] == 9:
                    X_original.append([x,y,z,factor*po[x][y][z]])
                    X_po.append(factor*[po[x][y][z]])
    
    if WHAT == 'cluster':
        return X_original, X_po
    
    if WHAT == 'cluster+infos':
        return X_original, X_po, shape, po

    if WHAT == 'simulation':
        optimizer = open_pop_from_pickle(ind_exp_name,199,PICKLE_LOCATION,returnAllOptimizer = True)
        return optimizer,ind, X_original
    
    if WHAT == 'clusterANDsimulation':
        optimizer = open_pop_from_pickle(ind_exp_name,199,PICKLE_LOCATION,returnAllOptimizer = True)

        return optimizer,ind, X_original, X_po



def open_clustering_optmization(EXP_NAME,ParamsSCAN,PC_NAME,COARSE_GRAIN,SELECT_TYPE,Inovation,RUN_DIR,SELECT_ROBOT,eps=None,eps_po=None):

    if Inovation == False:
        if ParamsSCAN:
            with open('/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/ClusterOptimization_{1}_withCaricaturesCheck{2}.pickle'.format(PC_NAME,EXP_NAME,SELECT_TYPE), 'rb') as handle:
                ClusterOptimization = pickle.load(handle)
        else:
            with open('/home/{3}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/ClusterOptimization_{0}_{1}-{2}_CG{4}.pickle'.format(EXP_NAME,eps,eps_po,PC_NAME,COARSE_GRAIN), 'rb') as handle:
                ClusterOptimization = pickle.load(handle)
    elif Inovation:
        with open('/home/{0}/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/ClusterOptWithCaricaturesCheck{1}_{2}-{3}.pickle'.format(PC_NAME,EXP_NAME,RUN_DIR,SELECT_ROBOT), 'rb') as handle:
            ClusterOptimization = pickle.load(handle)


    return ClusterOptimization
    

def plot_clustering_transference(SIZE,eps,eps_po):

    envs = []
    stiffs = []
    clusternumber = []
    original_envs = []
    original_stiffs = []
    functional_stiff = []
    differenceNClusters = []
    best_stiff = []
    fits = []

    for from_env in ['AQUA','EARTH','MARS']:
        EXP_NAME = 'Fixed_{1}_from{0}'.format(from_env,SIZE)
        print('Entered in {0}'.format(EXP_NAME))
        try:
            with open('/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/ClusterOptimization_{0}_{1}-{2}.pickle'.format(EXP_NAME,eps,eps_po), 'rb') as handle:
                ClusterOptimization = pickle.load(handle)

            with open('/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/Optimization_{0}.pickle'.format(EXP_NAME), 'rb') as handle:
                optimization_results = pickle.load(handle)
        
        except:
            print("The optimization results does not exist")
            return

        for count, tag in enumerate(optimization_results):
            for to_env in ['AQUA','EARTH','MARS']:
                for to_stiff in [4,5,6,7]:
                    try:
                        fit_value = optimization_results[tag]['to_{0}'.format(to_env)][to_stiff]
                        cluster_value = ClusterOptimization[tag]['to_{0}'.format(to_env)][to_stiff]['n_clusters']
                        clusterOriginal = ClusterOptimization[tag]['to_{0}'.format(optimization_results[tag]['originalEnv'])][optimization_results[tag]['originalStiff']]['n_clusters']
                        worked = True
                    except:
                        worked = False
                        print('problem in cluster {0}to{1} and to_stiff{2}- tag{3}'.format(from_env,to_env,to_stiff,tag))

                    if worked:
                        fits.append(fit_value)
                        clusternumber.append(cluster_value)
                        stiffs.append('5e+0{0}'.format(to_stiff))
                        envs.append(to_env)
                        original_envs.append(optimization_results[tag]['originalEnv'])
                        original_stiffs.append(optimization_results[tag]['originalStiff'])
                        differenceNClusters.append(clusterOriginal - cluster_value)

                        if (to_env == 'AQUA' and to_stiff == 5) or ((to_env == 'EARTH' or to_env == 'MARS') and to_stiff == 7):
                            best_stiff.append(True)
                        else:
                            best_stiff.append(False)
                        if (to_env == 'EARTH' or to_env == 'MARS') and (to_stiff == 4 or to_stiff == 5):
                            functional_stiff.append(False)
                        else:
                            functional_stiff.append(True)
                        
            if count%10 == 0:
                print('{0}/{1}'.format(count,len(optimization_results)))  

    plot_dict = {'Env':envs,'Stiff':stiffs,'FunctionalStiff':functional_stiff,'NumberOfClusters':clusternumber,
                        'OriginalEnv':original_envs,'OriginalStiff':original_stiffs,'BestStiff':best_stiff,
                    "Fit":fits, 'NumberOfClustersDifference':differenceNClusters
                    }

    plot_df = pd.DataFrame(plot_dict)
    print('Finishing processing data to plot')
    #return plot_df, count/3

    sns.set(font_scale = 1.3)
    c = sns.catplot(x="OriginalEnv", y="NumberOfClustersDifference", hue="Env", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df[plot_df.BestStiff == True],hue_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'])
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Just best stiff- Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/pointplot_DifferenceNumberofClusters_OriginalEnv_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    sns.set(font_scale = 1.3)
    c = sns.catplot(x="OriginalEnv", y="NumberOfClustersDifference", hue="Stiff", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df,col='Env',col_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/pointplot_DifferenceNumberofClusters_OriginalEnvStiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    sns.set(font_scale = 1.3)
    c = sns.catplot(x="Env", y="NumberOfClustersDifference", hue="Stiff", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df,col='OriginalEnv',col_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/pointplot_DifferenceNumberofClusters_EnvStiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()


    sns.set(font_scale = 1.3)
    c = sns.catplot(x="OriginalEnv", y="NumberOfClusters", hue="Stiff", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df,col='Env',col_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/pointplot_NumberofClusters_OriginalEnv_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    sns.set(font_scale = 1.3)
    c = sns.catplot(x="Env", y="NumberOfClusters", hue="Stiff", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df,col='OriginalEnv',col_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/point_plot_ClusterOptimizedChanging_EnvStiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    c = sns.catplot(x="Env", y="NumberOfClusters", hue="Stiff", data=plot_df,kind='box',col='OriginalEnv',sharey=False, #cut = 0,scale='count',
            palette="viridis_r",height=5, aspect=1.25,col_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/boxplot_plot_ClusterOptimizedChanging_EnvStiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    c = sns.catplot(x="Stiff", y="NumberOfClusters", hue="Env", data=plot_df,kind='point',col='OriginalEnv',sharey=False, #cut = 0,scale='count',
            palette="viridis_r",height=5, aspect=1.25,col_order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/point_plot_ClusterOptimizedChanging_Stiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    c = sns.catplot(x="Stiff", y="Fit", hue="Env", data=plot_df,kind='point',col='OriginalEnv',sharey=False, #cut = 0,scale='count',
            palette="viridis_r",height=5, aspect=1.25,col_order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/point_plot_ClusterOptimizedChanging_StiffXFit_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    c = sns.catplot(x="Env", y="Fit", hue="Stiff", data=plot_df,kind='point',col='OriginalEnv',sharey=False, #cut = 0,scale='count',
            palette="viridis_r",height=5, aspect=1.25,col_order = ['AQUA','MARS','EARTH'],order = ['AQUA','MARS','EARTH'], col_wrap=2)
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds evaluated - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/point_plot_ClusterOptimizedChanging_EnvXFit_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()

    c = sns.catplot(x="Env", y="NumberOfClusters", hue="OriginalEnv", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df[plot_df.FunctionalStiff == True],
                order = ['AQUA','MARS','EARTH'])
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds - Functional Stiff - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/point_plot_ClusterOptimizedChanging_EnvStiff_FunctionalStiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close()  

    c = sns.catplot(x="Stiff", y="NumberOfClusters", hue="Env", sharey=False,
                capsize=.15, palette="viridis_r", height=5, aspect=1.25,
                kind="point", data=plot_df[plot_df.FunctionalStiff == True])
    c.fig.subplots_adjust(top=0.85)
    c.fig.suptitle('Optimized Robots - {0}^3 - {1} inds - Functional Stiff - Eps{2}-EpsPo{3}'.format(SIZE,count,eps,eps_po), fontsize=15)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/FixedShapeOptimization/point_plot_ClusterOptimizedChanging_Stiff_FunctionalStiff_{0}^3_{1}-{2}.png".format(SIZE,eps,eps_po), bbox_inches='tight')
    plt.close() 
