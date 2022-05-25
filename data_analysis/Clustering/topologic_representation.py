import numpy as np
import pandas as pd
import pickle
import sys
import os
PC_NAME = 'renatabb'
from clustering_utils import open_cluster_seed, neighbor_voxels
from clustering_algorithm import processing_data_to_cluster_light_no_pt

def calc_cluster_info_usingGroundTouchTrace(clustering_X,X_original_data,TouchTrace,factor=10):
    """ Using the labels of the clustering process, calculate the CM position, mass and 
    mean phase offset of each cluster +
    using GroundTouchTrace, tag if each cluster touch ground using the info of the trace of its movement"""
    
    #TouchTrace[cluster_label] = {'Time':list(this_voxel_trace["Time"]),
        # 'XTrace':cluster_XTrace,'YTrace':cluster_YTrace ,'ZTrace':cluster_ZTrace
        #,'TouchGroundTrace':cluster_TouchGroundTrace}

    clusters_info = {}

    #clus_id is the number identifying the cluster (label)
    for clus_id in range(clustering_X['n_clusters']):
        this_cluster_touch_ground = False
        if np.sum(TouchTrace[clus_id]['TouchGroundTrace']) > 0.0:
            this_cluster_touch_ground = True

        clusters_info[clus_id] = {'meanX':[],'meanY':[],'meanZ':[],'meanPHASE':[],'numVOXEL':0,
                                    'touchGround':this_cluster_touch_ground,
                                    'minX':0,'minY':0,'minZ':0,'maxX':0,'maxY':0,'maxZ':0}

    for i in range(len(X_original_data)): #for each voxel in the body
        clus_id = clustering_X['labels'][i]
        
        clusters_info[clus_id]['meanX'].append(X_original_data[i][0])
        clusters_info[clus_id]['meanY'].append(X_original_data[i][1])
        clusters_info[clus_id]['meanZ'].append(X_original_data[i][2])
        clusters_info[clus_id]['meanPHASE'].append(X_original_data[i][3])
        clusters_info[clus_id]['numVOXEL'] += 1


    for clus_id in clusters_info:
        clusters_info[clus_id]['minX'] = np.min(clusters_info[clus_id]['meanX'])
        clusters_info[clus_id]['minY'] = np.min(clusters_info[clus_id]['meanY'])
        clusters_info[clus_id]['minZ'] = np.min(clusters_info[clus_id]['meanZ'])

        clusters_info[clus_id]['maxX'] = np.max(clusters_info[clus_id]['meanX'])
        clusters_info[clus_id]['maxY'] = np.max(clusters_info[clus_id]['meanY'])
        clusters_info[clus_id]['maxZ'] = np.max(clusters_info[clus_id]['meanZ'])

        clusters_info[clus_id]['meanX'] = round(np.mean(clusters_info[clus_id]['meanX']),2)
        clusters_info[clus_id]['meanY'] = round(np.mean(clusters_info[clus_id]['meanY']),2) 
        clusters_info[clus_id]['meanZ'] = round(np.mean(clusters_info[clus_id]['meanZ']),2) 
        clusters_info[clus_id]['meanPHASE'] = [round(np.mean(clusters_info[clus_id]['meanPHASE'])/factor,2),round(np.std(clusters_info[clus_id]['meanPHASE'])/factor,2)]

    return clusters_info

def calc_cluster_info(clustering_X,X_original_data):
    """ Using the labels of the clustering process, calculate the CM position, mass and 
    mean phase offset of each cluster """

    clusters_info = {}

    #clus_id is the number identifying the cluster (label)
    for clus_id in range(clustering_X['n_clusters']):
        clusters_info[clus_id] = {'meanX':[],'meanY':[],'meanZ':[],'meanPHASE':[],'numVOXEL':0,'touchGround':False,
                                    'minX':0,'minY':0,'minZ':0,'maxX':0,'maxY':0,'maxZ':0}

    SOME_CLUSTER_TOUCH_GROUND = False
    for i in range(len(X_original_data)): #for each voxel in the body
        clus_id = clustering_X['labels'][i]
        if X_original_data[i][2] == 0:                       # if this voxel (that belong to the clus_id), touch the ground (z=0)
            clusters_info[clus_id]['touchGround'] = True    #the cluster is classified as touching the ground
            SOME_CLUSTER_TOUCH_GROUND = True
        clusters_info[clus_id]['meanX'].append(X_original_data[i][0])
        clusters_info[clus_id]['meanY'].append(X_original_data[i][1])
        clusters_info[clus_id]['meanZ'].append(X_original_data[i][2])
        clusters_info[clus_id]['meanPHASE'].append(X_original_data[i][3])
        clusters_info[clus_id]['numVOXEL'] += 1

    ground = 0
    while SOME_CLUSTER_TOUCH_GROUND == False: #if there is no voxel touching the ground because the robots starts 'flying'
        ground += 1
        for i in range(len(X_original_data)): #for each voxel in the body
            clus_id = clustering_X['labels'][i]
            if X_original_data[i][2] == ground:
                clusters_info[clus_id]['touchGround'] = True
                SOME_CLUSTER_TOUCH_GROUND = True

    for clus_id in clusters_info:
        clusters_info[clus_id]['minX'] = np.min(clusters_info[clus_id]['meanX'])
        clusters_info[clus_id]['minY'] = np.min(clusters_info[clus_id]['meanY'])
        clusters_info[clus_id]['minZ'] = np.min(clusters_info[clus_id]['meanZ'])

        clusters_info[clus_id]['maxX'] = np.max(clusters_info[clus_id]['meanX'])
        clusters_info[clus_id]['maxY'] = np.max(clusters_info[clus_id]['meanY'])
        clusters_info[clus_id]['maxZ'] = np.max(clusters_info[clus_id]['meanZ'])

        clusters_info[clus_id]['meanX'] = round(np.mean(clusters_info[clus_id]['meanX']),2)
        clusters_info[clus_id]['meanY'] = round(np.mean(clusters_info[clus_id]['meanY']),2) 
        clusters_info[clus_id]['meanZ'] = round(np.mean(clusters_info[clus_id]['meanZ']),2) 
        clusters_info[clus_id]['meanPHASE'] = [round(np.mean(clusters_info[clus_id]['meanPHASE']),2),round(np.std(clusters_info[clus_id]['meanPHASE']),2)]

    return clusters_info


def calc_cluster_connections(clustering_X,X_original_data):
    """ Using the labels of the clustering process, check which clusters are connected with each other by
    comparing each voxel of each cluster """

    connection_matrix = np.full((clustering_X['n_clusters'],clustering_X['n_clusters']),None)
    connection_matrix = pd.DataFrame(connection_matrix)

    for column_index, column_items in connection_matrix.items(): #iterate over the columns
        for row_index in range(len(column_items)): #iterate over lines
            if row_index != column_index and column_items[row_index] == None: #it is not diagonal and the connection was not searched yet
                CONNECTED = False
                for i, clus_id_i in enumerate(clustering_X['labels']):
                    for j, clus_id_j in enumerate(clustering_X['labels']):
                        if clus_id_i == column_index and clus_id_j == row_index:
                            CONNECTED = neighbor_voxels(X_original_data[i],X_original_data[j])
                            
                        if CONNECTED == True:
                            break
                    if CONNECTED == True:
                        break
                connection_matrix[row_index][column_index] = CONNECTED
                connection_matrix[column_index][row_index] = CONNECTED

    return connection_matrix


def process_clusters(SEED_INIT,SEED_END,EXP_NAME,CLUSTERING_NAME,ENCODE,MAX_GEN,SIZE):
    """ Using the labels of the clustering process, calculate the CM position, mass and 
    mean phase offset of each cluster and the connection matrix between the clusters"""
    
    ProcessedClusterInfos_location = "exp_analysis/{0}/clustering/ProcessedClusterInfos".format(EXP_NAME)
    for seed in range(SEED_INIT,SEED_END+1):
        
        clustering_allX_results = open_cluster_seed(seed,EXP_NAME,CLUSTERING_NAME,encode=ENCODE)
        all_X = processing_data_to_cluster_light_no_pt(seed,EXP_NAME,SIZE,MAX_GEN,ENCODE,factor=1,return_gen = False)
        
        try:
            with open("~/locomotion_principles/data_analysis/{0}/ProcessedClusters_{1}_seed{2}.pickle".format(ProcessedClusterInfos_location,CLUSTERING_NAME,seed), 'rb') as handle:
                clusters_processed = pickle.load(handle)
        except:
            clusters_processed = {}

        for count, ind_id in enumerate(clustering_allX_results):
            if ind_id not in clusters_processed.keys():
                ind_clustering_info = calc_cluster_info(clustering_allX_results[ind_id],all_X[ind_id])
                connection_matrix = calc_cluster_connections(clustering_allX_results[ind_id],all_X[ind_id])

                clusters_processed[ind_id] = {'info':ind_clustering_info, 'connection':connection_matrix}
                
                #print('{0}/{1}'.format(count,len(clustering_allX_results)),ind_id)
                if count% 50 == 0:
                    with open("~/locomotion_principles/data_analysis/{0}/ProcessedClusters_{1}_seed{2}.pickle".format(ProcessedClusterInfos_location,CLUSTERING_NAME,seed), 'wb') as handle:
                        pickle.dump(clusters_processed, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("~/locomotion_principles/data_analysis/{0}/ProcessedClusters_{1}_seed{2}.pickle".format(ProcessedClusterInfos_location,CLUSTERING_NAME,seed), 'wb') as handle:
            pickle.dump(clusters_processed, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print ('finished saving seed {0}'.format(seed))


def process_clusters_Transference(SIZE,ParamsSCAN,COARSE_GRAIN,path,PC_NAME,SELECT_ROBOT,SELECT_TYPE,RUN_DIR,Inovation,FROM_ENV=['AQUA','MARS','EARTH'],EPS=None,EPS_PO=None):
    """ For the Transfered Robots:
    Using the labels of the clustering process, calculate the CM position, mass and 
    mean phase offset of each cluster and the connection matrix between the clusters"""
    
    sys.path.insert(0, os.path.abspath('/home/{0}/reconfigurable_organisms/data_analysis/TransferenceSelectedShapes/').format(PC_NAME))
    from cluster_TransferenceSelectedShape import open_clustering_optmization
    
    if Inovation == False:
        if SELECT_TYPE == 'OLD':
            SELECT_TYPE = ''
        elif SELECT_TYPE == 'BEST':
            SELECT_TYPE = '_BEST'
        elif SELECT_TYPE == 'BAD':
            SELECT_TYPE = '_WORST'
        elif SELECT_TYPE == 'WORST':
            SELECT_TYPE = '_WORSTallgen'
        

    for from_env in FROM_ENV:
        EXP_NAME = 'Final_{0}_{1}'.format(SIZE,from_env)
        clusters_processed = {}

        #Open All Infos of Clustered Transference Robots
        TransferedShapesClustersInfos = open_clustering_optmization(EXP_NAME,ParamsSCAN,PC_NAME,COARSE_GRAIN,SELECT_TYPE,Inovation,RUN_DIR,SELECT_ROBOT,eps=EPS,eps_po = EPS_PO)

        if Inovation == False:
            #Open ClustersMeanTrace
            with open("/home/{0}/reconfigurable_organisms/data_analysis/CMVoxelTrajectory/ClusterTraces/ClustersMeanTrace{1}{2}.pickle".format(PC_NAME,EXP_NAME,SELECT_TYPE), 'rb') as handle:
                ClustersMeanTrace = pickle.load(handle)
            #ClustersMeanTrace['{0}-{1}_{2}'.format(seed,shape,to_env)][cluster_label] = {'Time':list(this_voxel_trace["Time"]),
            # 'XTrace':cluster_XTrace,'YTrace':cluster_YTrace ,'ZTrace':cluster_ZTrace
            #,'TouchGroundTrace':cluster_TouchGroundTrace}
        elif Inovation:
            with open("/home/{0}/reconfigurable_organisms/data_analysis/CMVoxelTrajectory/ClusterTraces/ClustersMeanTrace{1}-{2}-{3}.pickle".format(PC_NAME,EXP_NAME,SELECT_ROBOT,RUN_DIR), 'rb') as handle:
                ClustersMeanTrace = pickle.load(handle)


        for ind_seed_shape in TransferedShapesClustersInfos:
            clusters_processed[ind_seed_shape] = {}
            seed = ind_seed_shape[ind_seed_shape.find('d')+1:ind_seed_shape.find('-')]
            shape = ind_seed_shape[ind_seed_shape.find('pe')+2:]
            for to_env in TransferedShapesClustersInfos[ind_seed_shape]:
                X_original = TransferedShapesClustersInfos[ind_seed_shape][to_env]['X_original']
                clustering_results = {k:v for k,v in TransferedShapesClustersInfos[ind_seed_shape][to_env].items() if k not in ['shape', 'X_original', 'po']}


                touch_trace = ClustersMeanTrace['{0}-{1}_{2}'.format(seed,shape,to_env)]
                ind_clustering_info = calc_cluster_info_usingGroundTouchTrace(clustering_results ,X_original,touch_trace,factor=10)
                connection_matrix = calc_cluster_connections(clustering_results ,X_original)

                clusters_processed[ind_seed_shape][to_env] = {'info':ind_clustering_info, 'connection':connection_matrix}
        
        if Inovation == False:  
            with open("{0}/ProcessedClusters_{1}{2}.pickle".format(path,EXP_NAME,SELECT_TYPE), 'wb') as handle:
                    pickle.dump(clusters_processed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif Inovation:
            with open("{0}/ProcessedClusters_{1}-{2}-{3}.pickle".format(path,EXP_NAME,SELECT_ROBOT,RUN_DIR), 'wb') as handle:
                pickle.dump(clusters_processed, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        print('Finished {0}'.format(EXP_NAME))

