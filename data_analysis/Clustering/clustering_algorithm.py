import pickle
import os
import sys
import numpy as np
from copy import deepcopy

from clustering_utils import save_cluster_seed, mean_po_per_cluster, replace_po_by_mean_po, phase_difference_inside_a_cluster


def dbscan(X,eps = 1.55, min_samples = 2, metric_types=['SC']):
    """
    Compute DBSCAN + optional performance evaluation
    """

    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    #Performance evaluation
    # metric = {}
    # for mtype in metric_types:
    #     if mtype == 'SC':
    #         try:
    #             metric[mtype] = metrics.silhouette_score(X, labels)
    #         except:
    #             metric[mtype] = None
    #     elif mtype == 'CH':
    #         metric[mtype] = metrics.calinski_harabasz_score(X, labels)
    #     elif mtype == 'DB':
    #         metric[mtype] = metrics.davies_bouldin_score(X, labels)
    #     else:
    #         metric[mtype] = 'non identified metric'
    
    return labels, n_clusters, n_noise #, metric

def look_for_similar_neighbors(X,labels,out_index):
    """
    Check if a outlier voxel is similar enough of some neighbor voxel of it
    """
    neighbors = {}
    X_outlier, Y_outlier, Z_outlier = X[out_index][0],X[out_index][1],X[out_index][2]
    for i in range(len(X)):
        if labels[i] != -1: #if the voxel is in a cluster
            if ((X[i][0],X[i][1],X[i][2]) == (X_outlier + 1, Y_outlier, Z_outlier)) or \
                ((X[i][0],X[i][1],X[i][2]) == (X_outlier - 1, Y_outlier, Z_outlier)) or \
                ((X[i][0],X[i][1],X[i][2]) == (X_outlier , Y_outlier + 1, Z_outlier)) or \
                ((X[i][0],X[i][1],X[i][2]) == (X_outlier, Y_outlier -1, Z_outlier)) or \
                ((X[i][0],X[i][1],X[i][2]) == (X_outlier, Y_outlier, Z_outlier + 1)) or \
                ((X[i][0],X[i][1],X[i][2]) == (X_outlier, Y_outlier, Z_outlier -1)): #if it is a direct neighbor
                #if (np.abs(X[out_index][3] - X[i][3]) <= 4 and (X[out_index][3]*X[i][3])>0) or (np.abs(X[out_index][3] - X[i][3]) <= 2) : #if the phases are similar
                if (np.abs(X[out_index][3] - X[i][3]) <= 3) : #if the phases are similar
                    neighbors[np.abs(X[out_index][3] - X[i][3])] = labels[i]
            
            elif ((X[i][0],X[i][1],X[i][2]) == (X_outlier + 1, Y_outlier+1, Z_outlier)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier + 1, Y_outlier-1, Z_outlier)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier + 1, Y_outlier, Z_outlier+1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier + 1, Y_outlier, Z_outlier-1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier -1 , Y_outlier+1, Z_outlier)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier -1 , Y_outlier-1, Z_outlier)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier -1 , Y_outlier, Z_outlier+1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier -1 , Y_outlier, Z_outlier-1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier , Y_outlier + 1, Z_outlier +1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier , Y_outlier + 1, Z_outlier -1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier , Y_outlier - 1, Z_outlier +1)) or \
            ((X[i][0],X[i][1],X[i][2]) == (X_outlier , Y_outlier - 1, Z_outlier -1)): #if it is a second direct neighbor
                if (np.abs(X[out_index][3] - X[i][3]) <= 3) : #if the phases are similar
                    neighbors[np.abs(X[out_index][3] - X[i][3])] = labels[i]

    if len(list(neighbors)) > 0:
        min = np.min(list(neighbors))
        labels[out_index] = neighbors[min]
    
    return labels


def dbscan_outliers(X,labels,eps, min_samples, metric_types=['SC']):
    from sklearn.cluster import DBSCAN
    """
    Try to group outliers in their neighbours, and add a new label to each one of them if it is not possible
    """
    
    X_outliers = []
    X_outliers_pos = []

    for i in range(len(labels)):
        if labels[i] == -1:
            labels = look_for_similar_neighbors(X,labels,i)
            if labels[i] == -1: #if did not found similar neighbors to cluster together
                X_outliers.append(X[i])
                X_outliers_pos.append(i)
    
    if len(X_outliers)>=1:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_outliers)
        labels_out = db.labels_
        list_new_labels = {}
        new_label = labels.max() + 1
        new_labels = labels.copy()
    
        for i in range(len(labels_out)):
            if labels_out[i] != -1: #new cluster found
                if labels_out[i] in list(list_new_labels):
                    j = X_outliers_pos[i]
                    new_labels[j] = list_new_labels[labels_out[i]]
                else:
                    j = X_outliers_pos[i]
                    new_labels[j] = new_label
                    list_new_labels[labels_out[i]] = new_label
                    new_label += 1
    
    else:
        new_labels = labels.copy()
    
     # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(new_labels)) - (1 if -1 in new_labels else 0)
    n_noise = list(new_labels).count(-1)
    
    
    return new_labels, n_clusters, n_noise #, metric



def core_clustering_algorithm(X_original,X_po,EPS,EPS_PO,MIN_SAMPLES,COARSE_GRAIN,cluster_results,ind_id,save_gen = [False,None],
            save_shape = [False,None],save_po = [False,None],save_X_original = False,saveEpsEpsPo = False,
            saveCaricatureFitness = [False,None],factor = 10):
    """ 
    The steps to cluster the voxels.
    
    eps_po: value of phase offset to the first filter that cluster the voxels just by the phase offset criteria (not consider the distance between voxel)
            if eps_po = 0, this first filter does not happen

    Coarse grain: If the algorithm clustered all voxels together, try a more sensible parameter in this robot

    """
    THRESHOLD = 0.2
    #First Clustering - it will cluster just the phase offset 
    if EPS_PO > 0:
        X = deepcopy(X_original)
        labels_po, n_clusters_po, n_noise = dbscan(X_po,EPS_PO,MIN_SAMPLES)
        used_eps_po = EPS_PO
        DO_AGAIN, TOTAL_DIF = phase_difference_inside_a_cluster(X_original,labels_po,n_clusters_po, threshold = THRESHOLD)
        best_param_so_far = EPS_PO
        if COARSE_GRAIN == True:
            if DO_AGAIN: # If the algorithm clusterized all together
            #try a more sensible parameter if there are a distance > 0.2 inside the robot (it should not be just one cluser)
                KEEP_COARSE_GRAIN = True
                eps_po_coarse_grain = EPS_PO - 0.001
                while KEEP_COARSE_GRAIN:
                    if eps_po_coarse_grain <= 0. or eps_po_coarse_grain < EPS_PO - 0.01:
                        break
                    else:
                        labels_po, n_clusters_po, n_noise = dbscan(X_po,eps_po_coarse_grain,MIN_SAMPLES)
                        used_eps_po = eps_po_coarse_grain

                        DO_AGAIN, NEW_TOTAL_DIF = phase_difference_inside_a_cluster(X_original,labels_po,n_clusters_po, threshold = THRESHOLD)
                        if NEW_TOTAL_DIF < TOTAL_DIF:
                            best_param_so_far = eps_po_coarse_grain
                            TOTAL_DIF = NEW_TOTAL_DIF

                        if DO_AGAIN:
                            eps_po_coarse_grain = eps_po_coarse_grain - 0.001        
                        else:
                            KEEP_COARSE_GRAIN = False
                            break
        
        if best_param_so_far != used_eps_po:
            labels_po, n_clusters_po, n_noise = dbscan(X_po,best_param_so_far,MIN_SAMPLES)
            used_eps_po = best_param_so_far


        clusters_po_mean = mean_po_per_cluster(X_po,labels_po) #associates with each label the correspondent mean phase offset value
        X = replace_po_by_mean_po(clusters_po_mean,X,labels_po,factor) #returns a data structure X with the mean phase offset (clusterized) instead of the original
    else:
        X = X_original
    
    THRESHOLD = 2
    #2 - Base clustering
    labels, n_clusters, n_noise = dbscan(X,EPS,MIN_SAMPLES)
    used_eps = EPS
    best_param_so_far = EPS
    DO_AGAIN, TOTAL_DIF = phase_difference_inside_a_cluster(X_original,labels,n_clusters, threshold = THRESHOLD)

    cluster_results[ind_id] = {'labels':labels, "n_clusters":n_clusters, "n_noise":n_noise}
    if save_gen[0]:
        cluster_results[ind_id]['gen'] = save_gen[1]
    if save_shape[0]:
        cluster_results[ind_id]['shape'] = save_shape[1]
    if save_po[0]:
        cluster_results[ind_id]['po'] = save_po[1]
    if save_X_original:
        cluster_results[ind_id]['X_original'] = X_original
    if saveEpsEpsPo:
        cluster_results[ind_id]['Eps-EpsPo'] = [used_eps,used_eps_po]
    if saveCaricatureFitness[0]:
        cluster_results[ind_id]['CaricatureFitness'] = saveCaricatureFitness[1]

    
    #3 - if the algorithm clustered all together, try a more sensible parameter eps
    if COARSE_GRAIN == True:
        if DO_AGAIN:
            KEEP_COARSE_GRAIN = True
            eps_coarse_grain = EPS - 0.01
            while KEEP_COARSE_GRAIN:
                if eps_coarse_grain <= 0. or eps_coarse_grain < EPS - 0.1:  
                    break
                else:
                    labels, n_clusters, n_noise= dbscan(X,eps_coarse_grain,MIN_SAMPLES)
                    used_eps = eps_coarse_grain

                    DO_AGAIN, NEW_TOTAL_DIF = phase_difference_inside_a_cluster(X_original,labels,n_clusters, threshold = THRESHOLD)
                    if NEW_TOTAL_DIF < TOTAL_DIF:
                        best_param_so_far = eps_coarse_grain
                        TOTAL_DIF = NEW_TOTAL_DIF
                    
                    if DO_AGAIN:
                        eps_coarse_grain = eps_coarse_grain - 0.01        
                    else:
                        KEEP_COARSE_GRAIN = False
                        break


    if best_param_so_far != used_eps:
        labels, n_clusters, n_noise = dbscan(X,best_param_so_far,MIN_SAMPLES)
        used_eps_po = best_param_so_far

    cluster_results[ind_id]['labels'] = labels
    cluster_results[ind_id]['n_clusters'] = n_clusters
    cluster_results[ind_id]['n_noise'] = n_noise
                       

    #4 - last clustering: the outliers of the second cluster are each one of then a cluster of 1 voxel or are grouped in neighbours and similar clusters (FOR THAT, Use X_original to compare the actual PO values of the neighbours)
    if cluster_results[ind_id]['n_noise'] >= 1: 
        min_samples_second = 1
        labels = cluster_results[ind_id]['labels']
        new_labels, n_clusters, n_noise = dbscan_outliers(X_original,labels,EPS,min_samples_second)
        cluster_results[ind_id]['labels'] = new_labels
        cluster_results[ind_id]['n_clusters'] = n_clusters
        cluster_results[ind_id]['n_noise'] = n_noise
        if saveEpsEpsPo:
            cluster_results[ind_id]['Eps-EpsPo'] = [used_eps,used_eps_po] 
    
    return cluster_results


def processing_data_to_cluster_light_no_pt(seed, EXP_NAME, SIZE,MAX_GEN,encode = "ASCII",factor=1,return_gen= False):
    """
    Returns all_X and maybe all_X_gen

    all_X -> all_X[ind] = [x,y,z,factor*po[x][y][z]]
    
    all_X_gen - > all_X_gen[ind] = gen
    """
    all_X = {}
    all_X_gen = {}

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
        
    for gen in all_gen_dict:
        if gen <= MAX_GEN:
            for ind in all_gen_dict[gen]:
                shape, po = all_gen_dict[gen][ind][1], all_gen_dict[gen][ind][2]
                X = []
                for x in range(SIZE[0]):
                    for y in range(SIZE[1]):
                        for z in range(SIZE[2]):
                            if shape[x][y][z] == 9:
                                X.append([x,y,z,factor*po[x][y][z]])

                all_X[ind] = X
                all_X_gen[ind] = gen
    
    if return_gen:
        return all_X, all_X_gen
    else:
        return all_X

def processing_data_to_cluster_just_po(seed, EXP_NAME, SIZE,MAX_GEN,encode = "ASCII",factor=1):

    all_X = {}

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
        
    for gen in all_gen_dict:
        if gen <= MAX_GEN:
            for ind in all_gen_dict[gen]:
                shape, po = all_gen_dict[gen][ind][1], all_gen_dict[gen][ind][2]
                X = []
                for x in range(SIZE[0]):
                    for y in range(SIZE[1]):
                        for z in range(SIZE[2]):
                            if shape[x][y][z] == 9:
                                X.append(factor*[po[x][y][z]])

                all_X[ind] = X
    
    return all_X


def algorithm_cluster_each_seed(SEED_INIT, SEED_END,eps,min_samples, EXP_NAME, SIZE,MAX_GEN,CLUSTERING_NAME,eps_po=0.0,factor= 10,COARSE_GRAIN = False,encode = "ASCII"):
    """ 
    The steps to cluster the voxels.
    
    eps_po: value of phase offset to the first filter that cluster the voxels just by the phase offset criteria (not consider the distance between voxel)
            if eps_po = 0, this first filter does not happen

    Coarse grain: If the algorithm clustered all voxels together, try a more sensible parameter in this robot

    """

    for seed in range(SEED_INIT, SEED_END+1):
        print ('entered in seed {0}'.format(seed))
        
        #Collect all data to do the First Clustering (just PhaseOffset) and the Second (both Phase and position in space)
        all_X_just_po = processing_data_to_cluster_just_po(seed,EXP_NAME,SIZE,MAX_GEN,encode,factor = 1)
        all_X, all_X_gen = processing_data_to_cluster_light_no_pt(seed,EXP_NAME,SIZE,MAX_GEN,encode,factor=factor,return_gen = True)

        cluster_results = {}

        for count, ind in enumerate(all_X):
            X_original = all_X[ind]
            X_po = all_X_just_po[ind]

            cluster_results = core_clustering_algorithm(X_original,X_po,eps,eps_po,min_samples,COARSE_GRAIN,cluster_results,ind,factor = factor,
                                save_gen = [True,all_X_gen[ind]],
                                save_shape = [False,None],
                                save_po = [False,None]
                                )
            if count%5000 == 0:
                print ('{0}/{1}'.format(count,len(all_X)))
    
        save = cluster_results
        save_cluster_seed(save,seed,EXP_NAME,CLUSTERING_NAME)


