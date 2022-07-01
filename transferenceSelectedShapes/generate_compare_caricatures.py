import hashlib
import os
import time
import random
import numpy as np
from copy import deepcopy
import pickle
import subprocess as sub
from data_analysis.basic_analysis_utils import return_shape_po_fitness
from data_analysis.Clustering.clustering_utils import replace_po_by_mean_po
from data_analysis.Clustering.clustering_algorithm import core_clustering_algorithm, processing_data_to_cluster_just_po, processing_data_to_cluster_light_no_pt
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

COMPUTER_NAME = 'renata'


if sys.version_info[0] < 3:
    sys.path.insert(0, os.path.abspath('/home/{0}/locomotion_principles/'.format(COMPUTER_NAME))) 
    from tools.read_write_voxelyze import write_voxelyze_file
    from softbot import Genotype, Phenotype

    MyGenotype = Genotype
    MyPhenotype = Phenotype
    ENCODE = 'ASCII'
else:
    ENCODE = 'latin1'

def process_data_to_plot_nclustersperVoxels_caricatures(save_dir,EXP_LIST,COARSE_GRAIN,encode):

    """Function to use after Caricatures_info pickles are ready. It process all the data for plot"""

    dist_caricatures = {'Nclusters/Nvoxels':[],'Nclusters':[],'Nvoxels':[],'eps_epsPO':[],'eps':[],'eps_po':[],
                            'OriginalFitSimmilarityNormalized':[],'Exp':[],
                            'CaricatureToOriginalFitDifference':[],'Gen':[], 'PercentOfOriginal':[]}

    for exp_name in EXP_LIST:
        print(exp_name)
        if os.path.isfile('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,exp_name,COARSE_GRAIN)) is True:
            with open('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,exp_name,COARSE_GRAIN), 'rb') as handle:
                if sys.version_info[0] < 3:
                    all_cluster_results_saved = cPickle.load(handle)
                else:
                    all_cluster_results_saved = pickle.load(handle,encoding = ENCODE)
        else:
            print('The cluster to caricature dict does not exist')
            return

        if os.path.isfile('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,exp_name,COARSE_GRAIN)) is True:
            with open('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,exp_name,COARSE_GRAIN), 'rb') as handle:
                if sys.version_info[0] < 3:
                    robots_caricatures_info = cPickle.load(handle)
                else:
                    robots_caricatures_info = pickle.load(handle,encoding = encode)
        else:
            print('The cluster to caricature dict does not exist')
            return

        ids_list = list(robots_caricatures_info)
        percentil25 = np.percentile(ids_list,25)
        percentil50 = np.percentile(ids_list,50)
        percentil75 = np.percentile(ids_list,75)
        # gen1 = []
        # gen2 = []
        # gen3 = []
        # gen4 = []
        
        print('Entered in',exp_name)
        for seed in all_cluster_results_saved:
            for eps in all_cluster_results_saved[seed]:
                if float(eps) >= 1.0:
                    for eps_po in all_cluster_results_saved[seed][eps]:
                        for ind in all_cluster_results_saved[seed][eps][eps_po]:
                            if ind in list(robots_caricatures_info):
                                if seed == robots_caricatures_info[ind]['Seed']:
                                    n_clusters = all_cluster_results_saved[seed][eps][eps_po][ind]['n_clusters']
                                    n_voxel = len(all_cluster_results_saved[seed][eps][eps_po][ind]['labels'])

                                    key = '{0}-{1}'.format(eps,eps_po)

                                    if key in list(robots_caricatures_info[ind]):

                                        if ind <= percentil25:
                                            gen = '0-25%'
                                            #gen1.append(ind)
                                        elif ind <= percentil50:
                                            gen = '25-50%'
                                            #gen2.append(ind)
                                        elif ind <= percentil75:
                                            gen = '50-75%'
                                            #gen3.append(ind)
                                        else:
                                            gen = '75-100%'
                                            #gen4.append(ind)

                                        try:
                                            original_fit = robots_caricatures_info[ind]['OriginalFit'][0]
                                            percent_of_original = 100*robots_caricatures_info[ind][key][0]/original_fit
                                            dist_normalized = 1- (original_fit - robots_caricatures_info[ind][key][0])/original_fit
                                            dist = (robots_caricatures_info[ind][key][0] - original_fit )
                                            dist_caricatures['CaricatureToOriginalFitDifference'].append(dist)
                                            dist_caricatures['OriginalFitSimmilarityNormalized'].append(dist_normalized)
                                            dist_caricatures['PercentOfOriginal'].append(percent_of_original)

                                            dist_caricatures['Gen'].append(gen)
                                            dist_caricatures['Nclusters/Nvoxels'].append(float(n_clusters)/n_voxel)
                                            dist_caricatures['Nvoxels'].append(n_voxel)
                                            dist_caricatures['Nclusters'].append(n_clusters)
                                            dist_caricatures['eps_epsPO'].append(key)
                                            dist_caricatures['eps'].append(eps)
                                            dist_caricatures['eps_po'].append(eps_po)
                                            dist_caricatures['Exp'].append(exp_name[8:])
                                        
                                        except:
                                            print('Does not have Original Fit',exp_name,seed,ind)
    data_caricature = pd.DataFrame(data = dist_caricatures)
    print('FINISH - generating data frames to box plots')
    # print(len(gen1),gen1[0:10],gen1[-10:])
    # print
    # print(len(gen2),gen2[0:10],gen2[-10:])
    # print 
    # print(len(gen3),gen3[0:10],gen3[-10:])
    # print
    # print(len(gen4),gen4[0:10],gen4[-10:])
    return data_caricature


def plots_study_parameters_caricatures(DIMENSION,data_caricature,alpha = 1,DC=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set(font_scale = 1.5)

    if DC == False:
        col_order = ['{}_AQUA'.format(DIMENSION),'{}_AQUA_FLUID'.format(DIMENSION),'{}_MARS'.format(DIMENSION),'{}_EARTH'.format(DIMENSION)]
        folder_plots_name = 'Plots'
        multiple = 9 #eps_po number of options
        multiple2 = 10 #eps number of options
    else:
        col_order = ['{}_AQUA_DirectEncode'.format(DIMENSION),'{}_MARS_DirectEncode'.format(DIMENSION),'{}_EARTH_DirectEncode'.format(DIMENSION)]
        folder_plots_name = 'PlotsDE'
        multiple = 9
        multiple2 = 10

    g = sns.catplot(x='eps_epsPO', y='PercentOfOriginal', col="Exp",  col_wrap=2, height=6,col_order = col_order,
                aspect=3,kind="point", data=data_caricature.sort_values(by="eps_epsPO"),sharey=False)
    g.set_xticklabels(rotation=90)
    for ax in g.axes.flat:
        ax.grid(True, axis='both')
        for i in range(0,multiple2):
            ax.axvline(multiple*i, ls='--', c='red')
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_PercentOfOriginal_mean.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - PercentOfOriginal Mean')

    g = sns.catplot(x='eps_epsPO', y='PercentOfOriginal', col="Exp",  col_wrap=2, height=6,col_order = col_order,
                aspect=3,kind="box", data=data_caricature.sort_values(by="eps_epsPO"),sharey=False)
    g.set_xticklabels(rotation=90)
    for ax in g.axes.flat:
        ax.grid(True, axis='both')
        for i in range(0,multiple2):
            ax.axvline(multiple*i, ls='--', c='red')
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_PercentOfOriginal_dist.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - PercentOfOriginal BoxPlot Distributions')

    g = sns.catplot(x='eps_epsPO', y='Nclusters/Nvoxels', col="Exp",  col_wrap=2, height=6,col_order = col_order,
                aspect=3,kind="point", data=data_caricature.sort_values(by="eps_epsPO"))
    g.set_xticklabels(rotation=90)
    for ax in g.axes.flat:
        ax.grid(True, axis='both')
        for i in range(0,multiple2):
            ax.axvline(multiple*i, ls='--', c='red')
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_NclustersperNvoxels_mean.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - Nclusters/Nvoxels mean')


    g = sns.catplot(x='eps_epsPO', y='Nclusters', col="Exp",col_wrap=2, height=6,col_order = col_order,
             aspect=3,kind="box",sharey=False, data=data_caricature.sort_values(by="eps_epsPO")) #violin, scale='count',cut=0
    g.set_xticklabels(rotation=90)
    for ax in g.axes.flat:
        ax.grid(True, axis='both')
        for i in range(0,multiple2):
            ax.axvline(multiple*i, ls='--', c='red')
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_Nclusters_dist.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - Nclusters distribution')


    g = sns.catplot(x='eps_epsPO', y='CaricatureToOriginalFitDifference', col="Exp", col_wrap=2, height=6,col_order = col_order,
                aspect=3,kind="box",sharey=False, data=data_caricature.sort_values(by="eps_epsPO"))
    g.set_xticklabels(rotation=90)
    for ax in g.axes.flat:
        ax.grid(True, axis='both')
        for i in range(0,multiple2):
            ax.axvline(multiple*i, ls='--', c='red')
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_CaricatureToOriginalDiff_dist.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - CaricatureToOriginalFitDifference boxplot distribution')

    g = sns.catplot(x='eps_epsPO', y='CaricatureToOriginalFitDifference', hue = 'Gen',col="Exp", col_wrap=2, height=6,col_order = col_order,
                aspect=3,kind="point", data=data_caricature.sort_values(by="eps_epsPO"),
                palette = sns.color_palette(['lightcoral','orange','darkseagreen','cornflowerblue']), #sns.color_palette("husl", 6),"Set2")
                hue_order = ['0-25%','25-50%','50-75%','75-100%'])
    g.set_xticklabels(rotation=90)
    for ax in g.axes.flat:
        ax.grid(True, axis='both')
        for i in range(0,multiple2):
            ax.axvline(multiple*i, ls='--', c='red')
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_CaricatureToOriginalDiff_mean_percentiles.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH -CaricatureToOriginalFitDifference mean percentiles')


    sns.lmplot(data=data_caricature,    x='Nvoxels', y="Nclusters",    height=5)
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_NvoxelsXnvoxels.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - x=Nvoxels, y=Nclusters')

    sns.jointplot(data=data_caricature,x='Nvoxels', y="Nclusters")
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/{1}/{0}_NvoxelsXnvoxels_joint.png".format(DIMENSION,folder_plots_name), bbox_inches='tight')
    plt.close()
    print('FINISH - x=Nvoxels, y=Nclusters jouintplot')




    


def process_data_to_plot_diff_of_caricatures(save_dir,EXP_NAME,COARSE_GRAIN):

    """Function to use after Caricatures_info pickles are ready. It process all the data for plot"""

    if os.path.isfile('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN)) is True:
        with open('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'rb') as handle:
            robots_caricatures_info = pickle.load(handle)
    
    else:
        print('Do not have this dict of caricatures')
        return
    dist_caricatures = {'OriginalFitSimmilarity':[],'eps_epsPO':[],'eps':[],'eps_po':[]}

    count_robots = 0
    for robot in robots_caricatures_info:
        try:
            original_fit = robots_caricatures_info[robot]['OriginalFit'][0]
            count_robots += 1
            for key in robots_caricatures_info[robot]:
                if key != 'OriginalFit' and key != 'Seed' and key != 'Gen':
                    dist = 1- (original_fit - robots_caricatures_info[robot][key][0])/original_fit

                    dist_caricatures['OriginalFitSimmilarity'].append(dist)
                    dist_caricatures['eps_epsPO'].append(key)
                    dist_caricatures['eps'].append(float(key[:key.find('-')]))
                    dist_caricatures['eps_po'].append(float(key[key.find('-')+1:]))
        except:
            print('Does not have OriginalFit',EXP_NAME,robot)

    data_caricature = pd.DataFrame(data = dist_caricatures)
    print('FINISH - generating data frames to box plots')
    return data_caricature,count_robots

def plot_difference_from_original(data_caricature,count_robots,TITLE_NAME,COARSE_GRAIN):
    """ Function that create two plots to annalyze the caricatures qualities"""

    sns.set(font_scale = 1.3)
    sns.catplot(x='eps', y='OriginalFitSimmilarity',hue='eps_po', capsize=.2, palette="viridis_r", height=5, aspect=2,
                    kind="point",plot_kws=dict(alpha=0.3), data=data_caricature)
    plt.title('{0} - {1} robots evaluated- CG{2}'.format(TITLE_NAME,count_robots,COARSE_GRAIN))
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/Plots/{0}_all_data_CG{1}_CaricatureSimilarity_eps.png".format(TITLE_NAME,COARSE_GRAIN), bbox_inches='tight')
    plt.close()

    sns.set(font_scale = 1.3)
    sns.catplot(x='eps_po', y='OriginalFitSimmilarity',hue='eps', capsize=.2, palette="viridis_r", height=5, aspect=2,
                    kind="point",plot_kws=dict(alpha=0.3), data=data_caricature)
    plt.title('{0} - {1} robots evaluated- CG{2}'.format(TITLE_NAME,count_robots,COARSE_GRAIN))
    plt.savefig("/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/Plots/{0}_all_data_CG{1}_CaricatureSimilarity_eps_po.png".format(TITLE_NAME,COARSE_GRAIN), bbox_inches='tight')



def cluster_selected_robots_in_a_seed(selected_robots_dict,seed,eps,min_samples, TITLE_NAME, SIZE,MAX_GEN,eps_po,factor= 10,COARSE_GRAIN = False,encode = "ASCII"):
    
    """Function called by the clustering_algorithm_caricatures() to call the core cluster algorithm.
    It is the clustering algorithm that returns a cluster_results dict"""

    all_X_just_po = processing_data_to_cluster_just_po(seed,TITLE_NAME,SIZE,MAX_GEN,COMPUTER_NAME,encode,factor = 1)
    all_X = processing_data_to_cluster_light_no_pt(seed,TITLE_NAME,SIZE,MAX_GEN,COMPUTER_NAME,encode,factor=factor)

    cluster_results = {}

    for ind in selected_robots_dict:
        X_original = all_X[ind]
        X_po = all_X_just_po[ind]

        cluster_results = core_clustering_algorithm(X_original,X_po,eps,eps_po,min_samples,COARSE_GRAIN,cluster_results,ind,save_gen = [False,None],save_shape = [False,None],save_po = [False,None],save_X_original = False,factor = 10)
                
    return cluster_results

def clustering_algorithm_caricatures(EPS_LIST,EPSPO_LIST,best_ids_fit,seed,TITLE_NAME, SIZE,MAX_GEN,PICKLE_LOCATION,EXP_NAME,factor,COARSE_GRAIN,ENCODE):
    
    """Function called by the main_algorithm_caricatures().
    It does a clustering only in the inds, eps, eps_po that were not clustered yet."""

    save_dir = "/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures"  # voxelyzeFiles
    
    if os.path.isfile('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN)) is True:
        with open('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'rb') as handle:
            if sys.version_info[0] < 3:
                all_cluster_results_saved = pickle.load(handle)
            else:
                all_cluster_results_saved = pickle.load(handle,encoding = ENCODE)
            
    try:
        all_clusters_results_this_seed = all_cluster_results_saved[seed]
    except:
        all_clusters_results_this_seed = {}

    print('{0} inds selected for beeing the best up to gen {1}'.format(len(best_ids_fit),MAX_GEN))
    for eps in EPS_LIST:
        if eps not in list(all_clusters_results_this_seed.keys()):
            all_clusters_results_this_seed[eps] = {}
        for eps_po in EPSPO_LIST:
            if eps_po not in list(all_clusters_results_this_seed[eps]):
                print('Entered in eps {0} eps_po {1} - all inds'.format(eps,eps_po))
                cluster_results = cluster_selected_robots_in_a_seed(best_ids_fit,seed,eps,2, TITLE_NAME, SIZE,MAX_GEN,eps_po,factor= factor,COARSE_GRAIN = COARSE_GRAIN,encode = ENCODE)
                all_clusters_results_this_seed[eps][eps_po] = cluster_results
            else:
                provisory_ids_list = deepcopy(best_ids_fit)
                for ind in best_ids_fit:
                    if ind in list(all_clusters_results_this_seed[eps][eps_po]):
                        #print('{2} of seed {3} already evaluated for {0}-{1}'.format(eps,eps_po,ind,seed))
                        provisory_ids_list.pop(ind)
                if len(provisory_ids_list)> 0:
                    print('Entered in eps {0} eps_po {1} - some inds'.format(eps,eps_po))
                    cluster_results = cluster_selected_robots_in_a_seed(provisory_ids_list,seed,eps,2, TITLE_NAME, SIZE,MAX_GEN,eps_po,factor= factor,COARSE_GRAIN = COARSE_GRAIN,encode = ENCODE)
                    for ind in cluster_results:
                        all_clusters_results_this_seed[eps][eps_po][ind] = cluster_results[ind]
    
    print('finished clustering')

    if os.path.isfile('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN)) is True:
        all_cluster_results_saved[seed] = all_clusters_results_this_seed
        with open('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'wb') as handle:
            pickle.dump(all_cluster_results_saved, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        all_cluster_results_saved = {}
        all_cluster_results_saved[seed] = all_clusters_results_this_seed
        with open('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'wb') as handle:
            pickle.dump(all_cluster_results_saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    time.sleep(5)
    print('finished SAVING clustering')
    return all_cluster_results_saved

def caricature_simulation_algorithm(seed,SIZE,MAX_GEN,PICKLE_LOCATION,EXP_NAME,TITLE_NAME,COARSE_GRAIN):
    """Function called by the main_algorithm_caricatures().
    It writes the voxelyze file and then simulate the ind,eps,eps_po caricature that was not avaliated yet"""

    save_dir = "/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures"

    if os.path.isfile('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN)) is True:
        with open('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'rb') as handle:
            all_clusters_to_caricature = pickle.load(handle)
    else:
        print('This Clusters_to_caricature_info does not exist')
        return

    if os.path.isfile('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN)) is True:
        with open('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'rb') as handle:
            robots_caricatures_info = pickle.load(handle)
    else:
        robots_caricatures_info = {}

    caricatures_in = {}
    for eps in all_clusters_to_caricature[seed]:
        for eps_po in all_clusters_to_caricature[seed][eps]:
            for ind in all_clusters_to_caricature[seed][eps][eps_po]:
                tag = '{0}-{1}'.format(eps,eps_po)
                if ind not in list(robots_caricatures_info):
                    if ind in list(caricatures_in):
                        caricatures_in[ind].append([eps,eps_po])
                    else:
                        caricatures_in[ind] = [[eps,eps_po]]
                elif (ind in list(robots_caricatures_info)) and (tag not in list(robots_caricatures_info[ind])):
                    if ind in list(caricatures_in):
                        caricatures_in[ind].append([eps,eps_po])
                    else:
                        caricatures_in[ind] = [[eps,eps_po]]
    
    list_inds = list(caricatures_in.keys())
    print ('Number of inds to do caricature {0}'.format(len(list_inds)))
    for ind_id_listed in list_inds:
        NOT_FOUND = True
        GEN_MAX = MAX_GEN 
        GEN_MIN = 0
        GEN = MAX_GEN
        while NOT_FOUND:
            print(GEN)
            PICKLE_DIR = '{0}/{1}/{1}_{2}'.format(PICKLE_LOCATION,EXP_NAME,seed)
            pickle_file = "{0}/pickledPops/Gen_{1}.pickle".format(PICKLE_DIR, GEN)
            with open(pickle_file, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = pickle.load(handle)

            pop = optimizer.pop
            examplar = np.max([int(pop[index].id) for index in range(len(pop))])

            for ind in pop:
                if ind.id == ind_id_listed:
                    id_corrected = '{0}'.format(ind.id).zfill(5)
                    NOT_FOUND = False
                    print('FOUND')
                    print('Entered in {0}, Gen{1}'.format(id_corrected,GEN))
                    my_env = optimizer.env[0]
                    if ind.id not in list(robots_caricatures_info):
                        print('Doing original sim')
                        robots_caricatures_info[ind.id] = {'Gen':GEN,'Seed':seed}
                        
                        #original
                        save_name = "{0}_seed{1}_gen{2}_original".format(EXP_NAME,seed,GEN)
                        write_voxelyze_file(optimizer.sim, my_env, ind, save_dir, save_name)
                        all_tag_keys = [('<normAbsoluteDisplacement>','OriginalFit')]

                        id_corrected = '{0}'.format(ind.id).zfill(5)
                        robot_vxa = '{2}/voxelyzeFiles/{0}--id_{1}.vxa'.format(save_name,id_corrected,save_dir)
                        robot_out = '/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/fitnessFiles/softbotsOutput--id_{0}.xml'.format(id_corrected)
                        if os.path.isfile(robot_out) is True:
                            sub.call("rm " + robot_out, shell=True)

                        sub.Popen("/home/renata/locomotion_principles/_voxcad/voxelyzeMain/voxelyze -f " + robot_vxa, shell=True)
                        time.sleep(5)
                        DO_NOT_GET_THIS = False
                        OUTPUT = False
                        count_redo = 0
                        while OUTPUT == False:
                            if os.path.isfile(robot_out) is True:
                                OUTPUT = True
                            else:
                                time.sleep(5)
                                count_redo += 1
                                print(count_redo)
                            if count_redo >= 200:
                                DO_NOT_GET_THIS = True
                                OUTPUT = True
                        
                        if DO_NOT_GET_THIS == False:
                            this_robot = open(robot_out)
                            for line in this_robot:
                                for tag, key in all_tag_keys:
                                    if tag in line:
                                        robots_caricatures_info[ind.id][key] = [float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])]
                        

                        sub.call("rm " + robot_out, shell=True)    
                        sub.call("rm " + robot_vxa, shell=True)

                    #caricatures
                    for i in range(0,len(caricatures_in[ind.id])):
                        id_corrected = '{0}'.format(ind.id).zfill(5)
                        eps,eps_po = caricatures_in[ind.id][i][0],caricatures_in[ind.id][i][1]
                        print('Doing caricatures of ind {2} with eps {0}, eps_po {1}'.format(eps,eps_po,id_corrected ))
                        all_tag_keys = [('<normAbsoluteDisplacement>','{0}-{1}'.format(eps,eps_po))]
                        
                        all_X = processing_data_to_cluster_light_no_pt(seed,TITLE_NAME,SIZE,MAX_GEN,COMPUTER_NAME,encode = 'ASCII',factor=10)
                        data = all_X[ind.id]
                        labels = all_clusters_to_caricature[seed][eps][eps_po][ind.id]['labels']

                        clusters_mean = mean_po_per_cluster_index(data,labels,3)
                        X = replace_po_by_mean_po(clusters_mean,data,labels,0.1)
                        po_matrix = np.array(list_to_matrix_po(labels,X,SIZE))
                        itemsdeepcopy = deepcopy(ind.genotype.to_phenotype_mapping.items())
                        itemsdeepcopy[1][1]['state'] = po_matrix

                        save_name = "{0}_seed{1}_gen{2}_caricature_{3}_{4}".format(EXP_NAME,seed,GEN,eps,eps_po)
                        
                        write_voxelyze_file_caricature(optimizer.sim, my_env, ind, save_dir, save_name, itemsdeepcopy)
                        
                        print('Simulating caricature eps{0},eps_po{1}'.format(eps,eps_po))

                        
                        robot_vxa = '{2}/voxelyzeFiles/{0}--id_{1}.vxa'.format(save_name,id_corrected,save_dir)
                        robot_out = '/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures/fitnessFiles/softbotsOutput--id_{0}.xml'.format(id_corrected)
                        if os.path.isfile(robot_out) is True:
                            sub.call("rm " + robot_out, shell=True)

                        sub.Popen("/home/renata/locomotion_principles/_voxcad/voxelyzeMain/voxelyze -f " + robot_vxa, shell=True)
                        time.sleep(5)

                        DO_NOT_GET_THIS = False
                        OUTPUT = False
                        count_redo = 0
                        while OUTPUT == False:
                            if os.path.isfile(robot_out) is True:
                                OUTPUT = True
                            else:
                                time.sleep(5)
                                count_redo += 1
                                print(count_redo)
                            if count_redo >= 200:
                                DO_NOT_GET_THIS = True
                                OUTPUT = True
                        
                        if DO_NOT_GET_THIS == False:
                            this_robot = open(robot_out)
                            for line in this_robot:
                                for tag, key in all_tag_keys:
                                    if tag in line:
                                        robots_caricatures_info[ind.id][key] = [float(line[line.find(tag) + len(tag):line.find("</" + tag[1:])])]
                        
                        sub.call("rm " + robot_out, shell=True)    
                        sub.call("rm " + robot_vxa, shell=True)

                        if i%4 == 0:
                            print('Saving caricatures info of robot {0}'.format(ind.id))
                            save_dir = "/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures"
                            with open('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'wb') as handle:
                                pickle.dump(robots_caricatures_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    list_inds.remove(ind.id)
                    print('More {0} robots'.format(len(list_inds)))

                    print('Saving caricatures info of  robot {0}'.format(ind.id))
                    save_dir = "/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures"
                    with open('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'wb') as handle:
                        pickle.dump(robots_caricatures_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if ind_id_listed <  examplar:
                GEN_MAX = GEN
                GEN = GEN_MAX - (GEN_MAX-GEN_MIN)/2
            elif ind_id_listed > examplar:
                GEN_MIN = GEN
                GEN = GEN_MIN + (GEN_MAX-GEN_MIN)/2
                

    print('Saving caricatures infos of this seed - {0}'.format(seed))
    save_dir = "/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures"
    with open('{0}/Caricatures_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'wb') as handle:
        pickle.dump(robots_caricatures_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return robots_caricatures_info

def main_algorithm_caricatures(EPS_LIST,EPSPO_LIST,SEED_INIT, SEED_END,TITLE_NAME, SIZE,MAX_GEN,PICKLE_LOCATION,EXP_NAME,ENCODE,factor,COARSE_GRAIN,NUMBER_TO_ANNALYZE_IN_EACH_SEED = 50,CLUSTER = True,SIM = False):
    "Algorithm that does all the steps: selecting ids, clustering and simulation"
    save_dir = "/home/renata/locomotion_principles/data_analysis/Clustering_check_caricatures"

    for seed in range(SEED_INIT, SEED_END+1):
        print('Entered in seed {0}, EXP {1}'.format(seed,TITLE_NAME))

        best_ids_fit = selection_of_best_of_all_seeds(NUMBER_TO_ANNALYZE_IN_EACH_SEED,seed,TITLE_NAME,MAX_GEN,ENCODE,EXACT = True)
        print('{0} inds selected'.format(len(best_ids_fit)))

        if CLUSTER:
            all_cluster_results_saved = clustering_algorithm_caricatures(EPS_LIST,EPSPO_LIST,best_ids_fit,seed,TITLE_NAME, SIZE,MAX_GEN,PICKLE_LOCATION,EXP_NAME,factor,COARSE_GRAIN,ENCODE)
        else:
            try:
                with open('{0}/Clusters_to_caricature_info_{1}_CG{2}.pickle'.format(save_dir,EXP_NAME,COARSE_GRAIN), 'rb') as handle:
                    all_cluster_results_saved = pickle.load(handle)
            except:
                print('This pickle does not exist')
                exit()

        if SIM:
            if sys.version_info[0] >= 3:
                print('You are in a python 3 version, it will not work')
            
            print("Entered in simulation")
            robots_caricatures_info = caricature_simulation_algorithm(seed,SIZE,MAX_GEN,PICKLE_LOCATION,EXP_NAME,TITLE_NAME,COARSE_GRAIN)

        print('Finished seed {0}'.format(seed))

    return


def select_ids(NUMBER_TO_ANNALYZE_IN_EACH_SEED,GEN,SEED,PICKLE_LOCATION,EXP_NAME,RETURN_ENV = False):

    list_of_ids = {}
    PICKLE_DIR = '{0}/{1}/{1}_{2}'.format(PICKLE_LOCATION,EXP_NAME,SEED)
    pickle_file = "{0}/pickledPops/Gen_{1}.pickle".format(PICKLE_DIR, GEN)
    with open(pickle_file, 'rb') as handle:
        [optimizer, random_state, numpy_random_state] = pickle.load(handle)
    pop = optimizer.pop

    for ind in pop:
        list_of_ids[float(ind.fitness)] = ind
    
    selected_ids = []
    for ind_included in sorted(list_of_ids)[-NUMBER_TO_ANNALYZE_IN_EACH_SEED:]:
        selected_ids.append(list_of_ids[ind_included])
    
    if RETURN_ENV:
        return selected_ids, optimizer.sim, optimizer.env[0]
    else:
        return selected_ids, optimizer.sim


def selection_of_best_of_all_seeds(NUMBER_TO_ANNALYZE_IN_EACH_SEED,SEED,TITLE_NAME,MAX_GEN,ENCODE,EXACT = False):

    all_fits_shape_po = return_shape_po_fitness(SEED,TITLE_NAME,MAX_GEN,COMPUTER_NAME,return_stiff = False,encode = ENCODE,exact = EXACT)

    best_fit_ids = {}
    best_ids_fit = {}
    for key_id in all_fits_shape_po:
        fit = all_fits_shape_po[key_id]['fit']

        if len(best_fit_ids) == NUMBER_TO_ANNALYZE_IN_EACH_SEED:
            if fit > sorted(best_fit_ids)[0]:
                best_fit_ids.pop(sorted(best_fit_ids)[0])
                best_fit_ids[fit] = key_id
        else:
            best_fit_ids[fit] = key_id
                
    for fit in best_fit_ids :
        best_ids_fit[best_fit_ids[fit]] = fit

    del(best_fit_ids)
    return best_ids_fit


def write_voxelyze_file_caricature(sim, env, individual, run_directory, run_name,itemsdeepcopy):
    """ Function almost like the original "write_voxelyze_file". 
    It has a small change to allow changing the phase offset state to a array not vinculated with genotype.tophenotypemapping("""

    # TODO: work in base.py to remove redundant static text in this function

    # obstacles: the following is used to freeze any elements not apart of the individual
    body_xlim = (0, individual.genotype.orig_size_xyz[0])
    body_ylim = (0, individual.genotype.orig_size_xyz[1])  # todo: if starting ind somewhere besides (0, 0)
    body_zlim = ((env.hurdle_height+1), individual.genotype.orig_size_xyz[2]+(env.hurdle_height+1))

    padding = env.num_hurdles * (env.space_between_hurdles + 1)
    x_pad = [padding, padding]
    y_pad = [padding, padding]

    if not env.circular_hurdles and env.num_hurdles > 0:
        if env.num_hurdles == 1:  # single hurdle
            y_pad = x_pad = [env.space_between_hurdles/2+1, env.space_between_hurdles/2+1]
        else:  # tunnel
            x_pad = [env.tunnel_width/2, env.tunnel_width/4]
            y_pad[0] = max(env.space_between_hurdles, body_ylim[1]-1) + 1 - body_ylim[1] + body_ylim[0]

    if env.forward_hurdles_only and env.num_hurdles > 0:  # ring
        y_pad[0] = body_ylim[1]/2

    if env.block_position > 0 and env.external_block:
        y_pad = x_pad = [0, env.block_position+1]

    workspace_xlim = (-x_pad[0], body_xlim[1] + x_pad[1])
    workspace_ylim = (-y_pad[0], body_ylim[1] + y_pad[1])
    workspace_zlim = (0, max(env.wall_height, body_zlim[1]))

    length_workspace_xyz = (float(workspace_xlim[1]-workspace_xlim[0]),
                            float(workspace_ylim[1]-workspace_ylim[0]),
                            float(workspace_zlim[1]-workspace_zlim[0]))

    fixed_regions_dict = {key: {"X": {}, "Y": {}, "dX": {}, "dY": {}} for key in range(4)}

    fixed_regions_dict[0] = {"X": 0, "dX": (x_pad[0]-1)/length_workspace_xyz[0]}

    fixed_regions_dict[1] = {"X": (body_xlim[1]-body_xlim[0]+x_pad[0]+1)/length_workspace_xyz[0],
                             "dX": 1 - (body_xlim[1]-body_xlim[0]+x_pad[0]+1)/length_workspace_xyz[0]}

    fixed_regions_dict[2] = {"Y": 0, "dY": (y_pad[0]-1)/length_workspace_xyz[1]}

    fixed_regions_dict[3] = {"Y": (body_ylim[1]-body_ylim[0]+y_pad[0]+1)/length_workspace_xyz[1],
                             "dY": 1 - (body_ylim[1]-body_ylim[0]+y_pad[0]+1)/length_workspace_xyz[1]}

    # update any env variables based on outputs instead of writing outputs in
    for name, details in individual.genotype.to_phenotype_mapping.items():
        if details["env_kws"] is not None:
            for env_key, env_func in details["env_kws"].items():
                setattr(env, env_key, env_func(details["state"]))  # currently only used when evolving frequency
                # print env_key, env_func(details["state"])

    voxelyze_file = open(run_directory + "/voxelyzeFiles/" + run_name + "--id_%05i.vxa" % individual.id, "w")

    voxelyze_file.write(
        "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n\
        <VXA Version=\"1.0\">\n\
        <Simulator>\n")

    # Sim
    for name, tag in sim.new_param_tag_dict.items():
        voxelyze_file.write(tag + str(getattr(sim, name)) + "</" + tag[1:] + "\n")

    voxelyze_file.write(
        "<Integration>\n\
        <Integrator>0</Integrator>\n\
        <DtFrac>" + str(sim.dt_frac) + "</DtFrac>\n\
        </Integration>\n\
        <Damping>\n\
        <BondDampingZ>1</BondDampingZ>\n\
        <ColDampingZ>0.8</ColDampingZ>\n\
        <SlowDampingZ>0.01</SlowDampingZ>\n\
        </Damping>\n\
        <Collisions>\n\
        <SelfColEnabled>" + str(int(sim.self_collisions_enabled)) + "</SelfColEnabled>\n\
        <ColSystem>3</ColSystem>\n\
        <CollisionHorizon>2</CollisionHorizon>\n\
        </Collisions>\n\
        <Features>\n\
        <FluidDampEnabled>0</FluidDampEnabled>\n\
        <PoissonKickBackEnabled>0</PoissonKickBackEnabled>\n\
        <EnforceLatticeEnabled>0</EnforceLatticeEnabled>\n\
        </Features>\n\
        <SurfMesh>\n\
        <CMesh>\n\
        <DrawSmooth>1</DrawSmooth>\n\
        <Vertices/>\n\
        <Facets/>\n\
        <Lines/>\n\
        </CMesh>\n\
        </SurfMesh>\n\
        <StopCondition>\n\
        <StopConditionType>" + str(int(sim.stop_condition)) + "</StopConditionType>\n\
        <StopConditionValue>" + str(sim.simulation_time) + "</StopConditionValue>\n\
        <InitCmTime>" + str(sim.fitness_eval_init_time) + "</InitCmTime>\n\
        <ActuationStartTime>" + str(sim.actuation_start_time) + "</ActuationStartTime>\n\
        </StopCondition>\n\
        <EquilibriumMode>\n\
        <EquilibriumModeEnabled>" + str(sim.equilibrium_mode) + "</EquilibriumModeEnabled>\n\
        </EquilibriumMode>\n\
        <GA>\n\
        <WriteFitnessFile>1</WriteFitnessFile>\n\
        <FitnessFileName>" + run_directory + "/fitnessFiles/softbotsOutput--id_%05i.xml" % individual.id +
        "</FitnessFileName>\n\
        <QhullTmpFile>" + run_directory + "/../_qhull/tempFiles/qhullInput--id_%05i.txt" % individual.id + "</QhullTmpFile>\n\
        <CurvaturesTmpFile>" + run_directory + "/../_qhull/tempFiles/curvatures--id_%05i.txt" % individual.id +
        "</CurvaturesTmpFile>\n\
        </GA>\n\
        <MinTempFact>" + str(sim.min_temp_fact) + "</MinTempFact>\n\
        <MaxTempFactChange>" + str(sim.max_temp_fact_change) + "</MaxTempFactChange>\n\
        <DampEvolvedStiffness>" + str(int(sim.damp_evolved_stiffness)) + "</DampEvolvedStiffness>\n\
        <MaxStiffnessChange>" + str(sim.max_stiffness_change) + "</MaxStiffnessChange>\n\
        <MinElasticMod>" + str(sim.min_elastic_mod) + "</MinElasticMod>\n\
        <MaxElasticMod>" + str(sim.max_elastic_mod) + "</MaxElasticMod>\n\
        <ErrorThreshold>" + str(0) + "</ErrorThreshold>\n\
        <ThresholdTime>" + str(0) + "</ThresholdTime>\n\
        <MaxKP>" + str(0) + "</MaxKP>\n\
        <MaxKI>" + str(0) + "</MaxKI>\n\
        <MaxANTIWINDUP>" + str(0) + "</MaxANTIWINDUP>\n")

    if hasattr(individual, "parent_lifetime"):
        if individual.parent_lifetime > 0:
            voxelyze_file.write("<ParentLifetime>" + str(individual.parent_lifetime) + "</ParentLifetime>\n")
        elif individual.lifetime > 0:
            voxelyze_file.write("<ParentLifetime>" + str(individual.lifetime) + "</ParentLifetime>\n")

    voxelyze_file.write("</Simulator>\n")

    # Env
    voxelyze_file.write(
        "<Environment>\n")
    for name, tag in env.new_param_tag_dict.items():
        voxelyze_file.write(tag + str(getattr(env, name)) + "</" + tag[1:] + "\n")

    if env.num_hurdles > 0:
        voxelyze_file.write(
            "<Boundary_Conditions>\n\
            <NumBCs>5</NumBCs>\n\
            <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>" + str(fixed_regions_dict[0]["X"]) + "</X>\n\
            <Y>0</Y>\n\
            <Z>0</Z>\n\
            <dX>" + str(fixed_regions_dict[0]["dX"]) + "</dX>\n\
            <dY>1</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
            </FRegion>\n\
            <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>" + str(fixed_regions_dict[1]["X"]) + "</X>\n\
            <Y>0</Y>\n\
            <Z>0</Z>\n\
            <dX>" + str(fixed_regions_dict[1]["dX"]) + "</dX>\n\
            <dY>1</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
            </FRegion>\n\
            <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>0</X>\n\
            <Y>" + str(fixed_regions_dict[2]["Y"]) + "</Y>\n\
            <Z>0</Z>\n\
            <dX>1</dX>\n\
            <dY>" + str(fixed_regions_dict[2]["dY"]) + "</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
            </FRegion>\n\
            <FRegion>\n\
            <PrimType>0</PrimType>\n\
            <X>0</X>\n\
            <Y>" + str(fixed_regions_dict[3]["Y"]) + "</Y>\n\
            <Z>0</Z>\n\
            <dX>1</dX>\n\
            <dY>" + str(fixed_regions_dict[3]["dY"]) + "</dY>\n\
            <dZ>1</dZ>\n\
            <Radius>0</Radius>\n\
            <R>0.4</R>\n\
            <G>0.6</G>\n\
            <B>0.4</B>\n\
            <alpha>1</alpha>\n\
            <DofFixed>63</DofFixed>\n\
            <ForceX>0</ForceX>\n\
            <ForceY>0</ForceY>\n\
            <ForceZ>0</ForceZ>\n\
            <TorqueX>0</TorqueX>\n\
            <TorqueY>0</TorqueY>\n\
            <TorqueZ>0</TorqueZ>\n\
            <DisplaceX>0</DisplaceX>\n\
            <DisplaceY>0</DisplaceY>\n\
            <DisplaceZ>0</DisplaceZ>\n\
            <AngDisplaceX>0</AngDisplaceX>\n\
            <AngDisplaceY>0</AngDisplaceY>\n\
            <AngDisplaceZ>0</AngDisplaceZ>\n\
            </FRegion>\n\
            <FRegion>\n\
                <PrimType>0</PrimType>\n\
                <X>0</X>\n\
                <Y>0</Y>\n\
                <Z>0</Z>\n\
                <dX>1</dX>\n\
                <dY>1</dY>\n\
                <dZ>" + str(env.hurdle_height/length_workspace_xyz[2]) + "</dZ>\n\
                <Radius>0</Radius>\n\
                <R>0.4</R>\n\
                <G>0.6</G>\n\
                <B>0.4</B>\n\
                <alpha>1</alpha>\n\
                <DofFixed>63</DofFixed>\n\
                <ForceX>0</ForceX>\n\
                <ForceY>0</ForceY>\n\
                <ForceZ>0</ForceZ>\n\
                <TorqueX>0</TorqueX>\n\
                <TorqueY>0</TorqueY>\n\
                <TorqueZ>0</TorqueZ>\n\
                <DisplaceX>0</DisplaceX>\n\
                <DisplaceY>0</DisplaceY>\n\
                <DisplaceZ>0</DisplaceZ>\n\
                <AngDisplaceX>0</AngDisplaceX>\n\
                <AngDisplaceY>0</AngDisplaceY>\n\
                <AngDisplaceZ>0</AngDisplaceZ>\n\
            </FRegion>\n\
            </Boundary_Conditions>\n"
        )

    else:
        voxelyze_file.write(
            "<Fixed_Regions>\n\
            <NumFixed>0</NumFixed>\n\
            </Fixed_Regions>\n\
            <Forced_Regions>\n\
            <NumForced>0</NumForced>\n\
            </Forced_Regions>\n"
            )

    voxelyze_file.write(
        "<Gravity>\n\
        <GravEnabled>" + str(env.gravity_enabled) + "</GravEnabled>\n\
        <GravAcc>" + str(env.grav_acc) + "</GravAcc>\n\
        <FloorEnabled>" + str(env.floor_enabled) + "</FloorEnabled>\n\
        <FloorSlope>" + str(env.floor_slope) + "</FloorSlope>\n\
        </Gravity>\n\
        <Thermal>\n\
        <TempEnabled>" + str(env.temp_enabled) + "</TempEnabled>\n\
        <TempAmp>" + str(env.temp_amp) + "</TempAmp>\n\
        <TempBase>25</TempBase>\n\
        <VaryTempEnabled>1</VaryTempEnabled>\n\
        <TempPeriod>" + str(1.0 / env.frequency) + "</TempPeriod>\n\
        </Thermal>\n\
        <LightSource>\n\
        <X>" + str(env.lightsource_xyz[0]) + "</X>\n\
        <Y>" + str(env.lightsource_xyz[1]) + "</Y>\n\
        <Z>" + str(env.lightsource_xyz[2]) + "</Z>\n\
        </LightSource>\n\
        <RegenerationModel>\n\
        <TiltVectorsUpdatesPerTempCycle>" + str(env.tilt_vectors_updates_per_cycle) + "</TiltVectorsUpdatesPerTempCycle>\n\
        <RegenerationModelUpdatesPerTempCycle>" + str(env.regeneration_model_updates_per_cycle) + "</RegenerationModelUpdatesPerTempCycle>\n\
        <NumHiddenRegenerationNeurons>" + str(env.num_hidden_regeneration_neurons) + "</NumHiddenRegenerationNeurons>\n\
        <RegenerationModelInputBias>" + str(int(env.regeneration_model_input_bias)) + "</RegenerationModelInputBias>\n\
        </RegenerationModel>\n\
        <ForwardModel>\n\
        <ForwardModelUpdatesPerTempCycle>" + str(env.forward_model_updates_per_cycle) + "</ForwardModelUpdatesPerTempCycle>\n\
        </ForwardModel>\n\
        <Controller>\n\
        <ControllerUpdatesPerTempCycle>" + str(env.controller_updates_per_cycle) + "</ControllerUpdatesPerTempCycle>\n\
        </Controller>\n\
        <Signaling>\n\
        <SignalingUpdatesPerTempCycle>" + str(env.signaling_updates_per_cycle) + "</SignalingUpdatesPerTempCycle>\n\
        <DepolarizationsPerTempCycle>" + str(env.depolarizations_per_cycle) + "</DepolarizationsPerTempCycle>\n\
        <RepolarizationsPerTempCycle>" + str(env.repolarizations_per_cycle) + "</RepolarizationsPerTempCycle>\n\
        </Signaling>\n\
        <GrowthAmplitude>" + str(env.growth_amp) + "</GrowthAmplitude>\n\
        <GrowthSpeedLimit>" + str(env.growth_speed_limit) + "</GrowthSpeedLimit>\n\
        <GreedyGrowth>" + str(int(env.greedy_growth)) + "</GreedyGrowth>\n\
        <GreedyThreshold>" + str(env.greedy_threshold) + "</GreedyThreshold>\n\
        <TimeBetweenTraces>" + str(env.time_between_traces) + "</TimeBetweenTraces>\n\
        <SavePassiveData>" + str(int(env.save_passive_data)) + "</SavePassiveData>\n\
        <StickyFloor>" + str(env.sticky_floor) + "</StickyFloor>\n\
        <BlockPushing>" + str(int(env.block_position > 0)) + "</BlockPushing>\n\
        <BlockMaterial>" + str(int(env.block_material)) + "</BlockMaterial>\n\
        <ContractOnly>" + str(int(env.contract_only)) + "</ContractOnly>\n\
        <ExpandOnly>" + str(int(env.expand_only)) + "</ExpandOnly>\n\
        <FallingProhibited>" + str(int(env.falling_prohibited)) + "</FallingProhibited>\n\
        <FluidEnvironment>" + str(int(env.fluid_environment)) + "</FluidEnvironment>\n\
        <AggregateDragCoefficient>" + str(int(env.aggregate_drag_coefficient)) + "</AggregateDragCoefficient>\n\
        </Environment>\n")

    #Defines the different material types in Palette
    #Material ID = 1 = fat (passivee)
    #Material ID = 2 = bone
    #Material ID = 3 = muscle with phase 0.01 (CTE)
    #Material ID = 4 = muscle with phase -0.01 (CTE)
    #And other types ...
    try:
        this_robot_stiffness = individual.genotype[2].feature
        string_for_md5 = str(this_robot_stiffness)
    except:
        this_robot_stiffness = env.muscle_stiffness
        string_for_md5 = "" 

    voxelyze_file.write(
        "<VXC Version=\"0.93\">\n\
        <Lattice>\n\
        <Lattice_Dim>" + str(env.lattice_dimension) + "</Lattice_Dim>\n\
        <X_Dim_Adj>1</X_Dim_Adj>\n\
        <Y_Dim_Adj>1</Y_Dim_Adj>\n\
        <Z_Dim_Adj>1</Z_Dim_Adj>\n\
        <X_Line_Offset>0</X_Line_Offset>\n\
        <Y_Line_Offset>0</Y_Line_Offset>\n\
        <X_Layer_Offset>0</X_Layer_Offset>\n\
        <Y_Layer_Offset>0</Y_Layer_Offset>\n\
        </Lattice>\n\
        <Voxel>\n\
        <Vox_Name>BOX</Vox_Name>\n\
        <X_Squeeze>1</X_Squeeze>\n\
        <Y_Squeeze>1</Y_Squeeze>\n\
        <Z_Squeeze>1</Z_Squeeze>\n\
        </Voxel>\n\
        <Palette>\n\
        <Material ID=\"1\">\n\
            <MatType>0</MatType>\n\
            <Name>Passive_Soft</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>1</Green>\n\
            <Blue>1</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.fat_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"2\">\n\
            <MatType>0</MatType>\n\
            <Name>Passive_Hard</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>0</Green>\n\
            <Blue>1</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.bone_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"3\">\n\
            <MatType>0</MatType>\n\
            <Name>Active_+</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>0</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"4\">\n\
            <MatType>0</MatType>\n\
            <Name>Active_-</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(-0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"5\">\n\
            <MatType>0</MatType>\n\
            <Name>Obstacle</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>0.784</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>5e+007</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"6\">\n\
            <MatType>0</MatType>\n\
            <Name>Head_Passive</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.fat_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"7\">\n\
            <MatType>0</MatType>\n\
            <Name>TouchSensor</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"8\">\n\
            <MatType>0</MatType>\n\
            <Name>Debris</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.block_density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>" + str(env.block_static_friction) + "</uStatic>\n\
            <uDynamic>" + str(env.block_dynamic_friction) + "</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"9\">\n\
            <MatType>0</MatType>\n\
            <Name>Active_+</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>0</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(this_robot_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>" + str(env.density) + "</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        </Palette>\n\
        <Structure Compression=\"ASCII_READABLE\">\n\
        <X_Voxels>" + str(length_workspace_xyz[0]) + "</X_Voxels>\n\
        <Y_Voxels>" + str(length_workspace_xyz[1]) + "</Y_Voxels>\n\
        <Z_Voxels>" + str(length_workspace_xyz[2]) + "</Z_Voxels>\n\
        <numRegenerationModelSynapses>" + str(env.num_regeneration_model_synapses) + "</numRegenerationModelSynapses>\n\
        <numForwardModelSynapses>" + str(env.num_forward_model_synapses) + "</numForwardModelSynapses>\n\
        <numControllerSynapses>" + str(env.num_controller_synapses) + "</numControllerSynapses>\n")

    all_tags = [details["tag"] for name, details in itemsdeepcopy]
    if "<Data>" not in all_tags:  # not evolving topology -- fixed presence/absence of voxels
        voxelyze_file.write("<Data>\n")
        for z in range(*workspace_zlim):
            voxelyze_file.write("<Layer><![CDATA[")
            for y in range(*workspace_ylim):
                for x in range(*workspace_xlim):

                    if (body_xlim[0] <= x < body_xlim[1]) and (body_ylim[0] <= y < body_ylim[1]) and (body_zlim[0] <= z < body_zlim[1]):

                        if env.biped and (z < body_zlim[1]*env.biped_leg_proportion) and (x == body_xlim[1]/2):
                            voxelyze_file.write("0")

                        elif env.falling_prohibited and z == body_zlim[1]-1:
                            voxelyze_file.write("6")  # head id

                        # elif env.kramer_fabric and (x > body_xlim[1]*.6 or x < body_xlim[1]*.3):
                        #     voxelyze_file.write("1")

                        if env.passive_body_only:
                            voxelyze_file.write("1")

                        else:
                            voxelyze_file.write("3")

                    elif env.block_position > 0 and env.external_block:
                        if (x == workspace_xlim[1]-1) and (y == workspace_ylim[1]-1) and (z == 0):
                            voxelyze_file.write("8")  # food
                        else:
                            voxelyze_file.write("0")

                    elif env.num_hurdles > 0:
                        # within the fixed regions
                        xy_centered = [x-body_xlim[1]/2, y-body_ylim[1]/2]
                        is_obstacle = False

                        if env.circular_hurdles:  # rings of circles
                            for hurdle in range(-1, env.num_hurdles + 1):
                                hurdle_radius = hurdle * env.space_between_hurdles
                                if abs(xy_centered[0]**2+xy_centered[1]**2-hurdle_radius**2) <= hurdle_radius:
                                    if z < env.hurdle_height:
                                        is_obstacle = True
                                        if env.debris and x % 2 == 0:
                                            is_obstacle = False

                                elif y == workspace_ylim[0] and env.back_stop and abs(xy_centered[0]) >= hurdle_radius/hurdle and abs(xy_centered[0]) <= hurdle_radius:
                                    if z < env.wall_height:
                                        if (env.fence and (x+z) % 2 == 0) or not env.fence:
                                            is_obstacle = True  # back wall

                        else:  # tunnel

                            start = body_ylim[1]*env.squeeze_start
                            end = body_ylim[1]*env.squeeze_end
                            p = (y-start) / float(end-start)

                            adj = 0
                            if y > body_ylim[1]*env.squeeze_start:
                                adj = int(p * env.squeeze_rate)

                            if env.constant_squeeze and y > body_ylim[1]*env.squeeze_end:
                                adj = min(int(env.squeeze_rate), workspace_xlim[1]-body_xlim[1])

                            wall = [workspace_xlim[0] + adj,
                                    workspace_xlim[1] - 1 - adj]

                            if x in wall and z < env.wall_height:
                                if (env.fence and (y+z) % 2 == 0) or not env.fence:
                                    is_obstacle = True  # wall

                            elif y % env.space_between_hurdles == 0 and z < env.hurdle_height:
                                is_obstacle = True  # hurdle
                                if env.debris and y > env.debris_start*body_ylim[1]:
                                    if (y % 2 == 0 and (x+z) % 2 == 0) or (y % 2 == 1 and (x+z) % 2 == 1) or x <= wall[0] + 1 or x >= wall[1] - 1:
                                        is_obstacle = False  # nothing
                                elif x <= wall[0] or x >= wall[1]:
                                    is_obstacle = False  # nothing

                                if y > env.hurdle_stop*body_ylim[1]:
                                    is_obstacle = False

                            if y == workspace_ylim[0] and env.back_stop and z < env.wall_height:
                                if (env.fence and (x+z) % 2 == 0) or not env.fence:
                                    is_obstacle = True  # back wall

                        if is_obstacle:
                            voxelyze_file.write("5")
                        else:
                            voxelyze_file.write("0")  # flat ground

                    else:
                        voxelyze_file.write("0")  # flat ground

            voxelyze_file.write("]]></Layer>\n")
        voxelyze_file.write("</Data>\n")

    # append custom parameters
    #string_for_md5 = "" moved to the stifness part

    for name, details in itemsdeepcopy:
        if ("Synapse" not in details["tag"]) and ("FakeTag" not in details["tag"]):
            # start tag
            if details["env_kws"] is None:
                voxelyze_file.write(details["tag"]+"\n")

            # record any additional params associated with the output
            if details["params"] is not None:
                for param_tag, param in zip(details["param_tags"], details["params"]):
                    voxelyze_file.write(param_tag + str(param) + "</" + param_tag[1:] + "\n")

            if details["env_kws"] is None:
                # write the output state matrix to file
                for z in range(*workspace_zlim):
                    voxelyze_file.write("<Layer><![CDATA[")
                    for y in range(*workspace_ylim):
                        for x in range(*workspace_xlim):

                            if (body_xlim[0] <= x < body_xlim[1]) and (body_ylim[0] <= y < body_ylim[1]) and (body_zlim[0] <= z < body_zlim[1]):
                                if individual.age == 0 and details["age_zero_overwrite"] is not None:
                                    state = details["age_zero_overwrite"]

                                elif details["switch_proportion"] > 0 and (x < body_xlim[1]-details["switch_proportion"]):
                                    # this is like the 'inverse' switch- if true then not switch and equal to other net
                                    switch_net_key = details["switch_name"]
                                    switch_net = individual.genotype.to_phenotype_mapping[switch_net_key]
                                    state = details["output_type"](switch_net["state"][x-body_xlim[0], y-body_ylim[0], z-(env.hurdle_height+1)])

                                else:
                                    state = details["output_type"](details["state"][x-body_xlim[0], y-body_ylim[0], z-(env.hurdle_height+1)])

                            # elif (env.block_position > 0) and (x == workspace_xlim[1] - 1) and (y == workspace_ylim[1] - 1) and (z == 0):
                            #     state = -1  # tiny food

                            elif env.circular_hurdles and z < env.hurdle_height and env.debris and x % 2 != 0:
                                state = 0
                                xy_centered = [x-body_xlim[1]/2, y-body_ylim[1]/2]
                                for hurdle in range(1, env.num_hurdles + 1):
                                    hurdle_radius = hurdle * env.space_between_hurdles
                                    if abs(xy_centered[0]**2 + xy_centered[1]**2 - hurdle_radius**2) <= hurdle_radius:
                                        state = -1  # tiny debris

                            elif env.num_hurdles > 0 and z < env.hurdle_height and workspace_xlim[0] < x < workspace_xlim[1]-1:
                                if env.debris_size < -1:
                                    state = 0.5*random.random()-1
                                else:
                                    state = env.debris_size  # tiny debris

                            elif env.block_position > 0 and env.external_block:
                                if (x >= workspace_xlim[1]-2) and (y >= workspace_ylim[1]-2) and (z < 2):
                                    state = env.block_material
                                else:
                                    state = 0

                            else:
                                state = 0

                            if env.num_hurdles > 0:
                                adj = int(np.sin(y/float(body_ylim[1]/1.5))*body_ylim[1]/1.5+body_ylim[1]/1.5-1)
                                wall = [workspace_xlim[0] + adj, adj+int(body_xlim[1]/1.5)-2]

                                if y < body_ylim[1]+3:
                                    adj = 0
                                    wall = [workspace_xlim[0]+body_xlim[1]/2+1, workspace_xlim[1]-1]

                                if y == body_ylim[1]+2 and (workspace_xlim[0]+body_xlim[1]/2 < x < body_xlim[0]) and z < env.wall_height:
                                    state = 5  # wall

                                if x in wall and z < env.wall_height:
                                    state = 5

                                if y == workspace_ylim[0] and env.back_stop and z < env.wall_height:
                                    if x > wall[0]:
                                        state = 5  # back wall

                            voxelyze_file.write(str(state))
                            if details["tag"] != "<Data>":  # TODO more dynamic
                                voxelyze_file.write(", ")
                            string_for_md5 += str(state)

                    voxelyze_file.write("]]></Layer>\n")

            # end tag
            if details["env_kws"] is None:
                voxelyze_file.write("</" + details["tag"][1:] + "\n")

    # special for SynapticWeights
    forward_model_layers = []
    forward_model_names = []
    controller_layers = []
    controller_names = []
    regeneration_model_layers = []
    regeneration_model_names = []
    using_forward_model = 0
    using_neural_controller = 0
    using_regeneration_model = 0
    for name, details in individual.genotype.to_phenotype_mapping.items():
        if details["tag"] == "<ForwardModelSynapseWeights>":
            forward_model_layers += [details["state"]]
            forward_model_names += [name]
            using_forward_model = 1

        elif details["tag"] == "<ControllerSynapseWeights>":
            controller_layers += [details["state"]]
            controller_names += [name]
            using_neural_controller = 1

        elif details["tag"] == "<RegenerationModelSynapseWeights>":
            regeneration_model_layers += [details["state"]]
            regeneration_model_names += [name]
            using_regeneration_model = 1

    sorted_forward_model_layers = [x for _, x in sorted(zip(forward_model_names, forward_model_layers))]
    sorted_controller_layers = [x for _, x in sorted(zip(controller_names, controller_layers))]
    sorted_regeneration_model_layers = [x for _, x in sorted(zip(regeneration_model_names, regeneration_model_layers))]

    # if copying a single direct encoding into each voxel
    num_vox = np.product(individual.genotype.orig_size_xyz)
    sx, sy, sz = individual.genotype.orig_size_xyz

    if len(sorted_forward_model_layers) == 1:
        sl = env.num_forward_model_synapses
        sorted_forward_model_layers = np.repeat(sorted_forward_model_layers[0], num_vox).reshape(sl, sx, sy, sz)

    if len(sorted_controller_layers) == 1:
        sl = env.num_controller_synapses
        sorted_controller_layers = np.repeat(sorted_controller_layers[0], num_vox).reshape(sl, sx, sy, sz)

    if len(sorted_regeneration_model_layers) == 1:
        sl = env.num_regeneration_model_synapses
        sorted_regeneration_model_layers = np.repeat(sorted_regeneration_model_layers[0], num_vox).reshape(sl, sx, sy, sz)
        # print np.var(sorted_regeneration_model_layers[0, :, :, :])

    synapse_tags = []
    synapse_layers = []
    if using_forward_model:
        synapse_tags += ["<ForwardModelSynapseWeights>"]
        synapse_layers += [sorted_forward_model_layers]
    if using_neural_controller:
        synapse_tags += ["<ControllerSynapseWeights>"]
        synapse_layers += [sorted_controller_layers]
    if using_regeneration_model:
        synapse_tags += ["<RegenerationModelSynapseWeights>"]
        synapse_layers += [sorted_regeneration_model_layers]

    if synapse_tags > 0:
        for this_tag, these_layers in zip(synapse_tags, synapse_layers):
            voxelyze_file.write(this_tag + "\n")
            for z in range(*workspace_zlim):
                voxelyze_file.write("<Layer><![CDATA[")
                for y in range(*workspace_ylim):
                    for x in range(*workspace_xlim):
                        for this_layer in these_layers:

                            if (body_xlim[0] <= x < body_xlim[1]) and (body_ylim[0] <= y < body_ylim[1]) and (body_zlim[0] <= z < body_zlim[1]):

                                state = float(this_layer[x-body_xlim[0], y-body_ylim[0], z-(env.hurdle_height+1)])

                            elif env.circular_hurdles and z < env.hurdle_height and env.debris and x % 2 != 0:
                                state = 0
                                xy_centered = [x-body_xlim[1]/2, y-body_ylim[1]/2]
                                for hurdle in range(1, env.num_hurdles + 1):
                                    hurdle_radius = hurdle * env.space_between_hurdles
                                    if abs(xy_centered[0]**2 + xy_centered[1]**2 - hurdle_radius**2) <= hurdle_radius:
                                        state = -1  # tiny debris

                            elif env.num_hurdles > 0 and z < env.hurdle_height and workspace_xlim[0] < x < workspace_xlim[1]-1:
                                if env.debris_size < -1:
                                    state = 0.5*random.random()-1
                                else:
                                    state = env.debris_size  # tiny debris

                            else:
                                state = 0

                            voxelyze_file.write(str(state))
                            voxelyze_file.write(", ")
                            string_for_md5 += str(state)

                voxelyze_file.write("]]></Layer>\n")

            # end tag
            voxelyze_file.write("</" + this_tag[1:] + "\n")

    voxelyze_file.write(
        "</Structure>\n\
        </VXC>\n\
        </VXA>")
    voxelyze_file.close()

    m = hashlib.md5()
    m.update(string_for_md5)

    return m.hexdigest()

def mean_po_per_cluster_index(X_original,labels,index=0):
    clusters_po_mean = {}

    for i, label in enumerate(labels):
        if label in list(clusters_po_mean):
            clusters_po_mean[label].append(X_original[i][index])
        else:
            clusters_po_mean[label] = [X_original[i][index]]
    
    for label in clusters_po_mean:
        clusters_po_mean[label] = np.mean(clusters_po_mean[label])
         

    return clusters_po_mean


def list_to_matrix_po(labels,X,size):

    matrix = np.full_like(np.zeros(size),0)
    for i in range(len(labels)):
        x,y,z,value = X[i][0], X[i][1],X[i][2],X[i][3]
        matrix[x][y][z] = value
        
    return matrix

def replace_po_by_mean_po(clusters_po_mean,X,labels,factor):

    for i in range(len(X)):
        #print(factor,clusters_po_mean[labels[i]])
        X[i][3] = factor*clusters_po_mean[labels[i]]

    return X

