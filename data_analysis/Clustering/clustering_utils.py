
import pickle
import numpy as np


def phase_difference_inside_a_cluster(X_original,labels,n_clusters, threshold):
    """
    Check if the phase difference inside one cluster is less than the threshold (2 -> 0.2 = 10% of the range of values)
    """
        
    clusters = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label in list(clusters):
                clusters[label].append(X_original[i][3])
            else:
                clusters[label] = [X_original[i][3]]
    
    DO_AGAIN = False
    TOTAL_DIF = 0
    for label in clusters:
        dif_this_clus = np.abs(np.max(clusters[label]) - np.min(clusters[label]))
        TOTAL_DIF += dif_this_clus
        if  dif_this_clus > threshold:
            DO_AGAIN = True

    if n_clusters <= 1:
        DO_AGAIN = True
        if n_clusters == 0:
            TOTAL_DIF = 10000
        else:
            if dif_this_clus > 0.1:
                TOTAL_DIF = dif_this_clus*10

    return DO_AGAIN, TOTAL_DIF
    
def replace_po_by_mean_po(clusters_po_mean,X,labels,factor):
    """ Function that returns a data structure X with the mean phase offset instead of the original phase offset"""
    for i in range(len(X)):
        if labels[i] != -1:
            X[i][3] = factor*clusters_po_mean[labels[i]]
    return X

def mean_po_per_cluster(X_po,labels):
    """ Function that associated with each label the correspondent mean phase offset value"""
    clusters_po_mean = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label in list(clusters_po_mean):
                clusters_po_mean[label].append(X_po[i][0])
            else:
                clusters_po_mean[label] = [X_po[i][0]]
    
    for label in clusters_po_mean:
        clusters_po_mean[label] = np.mean(clusters_po_mean[label])
    
    return clusters_po_mean

def neighbor_voxels(voxel_i,voxel_j):
    "Test if voxel_i(xi,yi,zi) and voxel_j(xj,yj,zj) are neighbors"
    x = np.abs(voxel_i[0] - voxel_j[0]) #(xi-xj)
    y = np.abs(voxel_i[1] - voxel_j[1]) #(yi-yj)
    z = np.abs(voxel_i[2] - voxel_j[2]) #(zi-zj)

    if (x + y + z) == 1: #if they are directed neighbors (connected), they will differ just in one (ki-kj) = 1 and the other will be zero
        return True                     #example: x=0, y=0 and z=1
    else:
        return False

def open_cluster_seed(seed,EXP_NAME,CLUSTERING_NAME,encode = "ASCII",CLUSTERING_FOLDER_NAME ='clustering'):
    """
    Returns clustering_allX_results

    clustering_allX_results[ind] = keys: 'labels','n_clusters', 'n_noise', 'gen'
    """
    with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/{1}/seeds_clustered_{2}/cluster_{2}_seed_{3}.pickle".format(EXP_NAME,CLUSTERING_FOLDER_NAME,CLUSTERING_NAME,seed), 'rb') as handle:
        if encode == "ASCII":
            clustering_allX_results = pickle.load(handle)
        else:
            clustering_allX_results = pickle.load(handle,encoding=encode)
        
    print ('finished recovering seed {0} cluster results'.format(seed))
    
    return clustering_allX_results


def save_cluster_seed(clustering_allX_results,seed,EXP_NAME,CLUSTERING_NAME,CLUSTERING_FOLDER_NAME ='clustering'):
    """Function to save the results of cluster one seed
    """

    with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/{1}/seeds_clustered_{2}/cluster_{2}_seed_{3}.pickle".format(EXP_NAME,CLUSTERING_FOLDER_NAME,CLUSTERING_NAME,seed), 'wb') as handle:
        pickle.dump(clustering_allX_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print ('finished saving seed {0}'.format(seed))