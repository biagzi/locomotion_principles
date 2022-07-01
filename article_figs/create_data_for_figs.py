import numpy as np
import pickle
import os
import sys
COMPUTER_NAME = '/home/renata'
sys.path.append('{0}/locomotion_principles/'.format(COMPUTER_NAME))
from data_analysis.Clustering.clustering_utils import open_cluster_seed
from data_analysis.Clustering.topologic_analysis import TopologyCalcs

def return_fit_stiff_nclusters(seed,EXP_NAME,MAX_GEN,CLUSTERING_NAME,encode = 'latin1'):
    """
    Returns all_fits_shape_po[ind] = {'fit','shape','po'}

    if exact = True: all_fits_shape_po[ind] =  {'fit','shape','po','stiff'}
    """


    all_fit = []
    all_stiff = []
    all_nclusters = []
    all_Nclusters_voxels = []

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
    
    clustering_allX_results = open_cluster_seed(seed,EXP_NAME,CLUSTERING_NAME)

    for gen in all_gen_dict:
        if gen <= MAX_GEN:
            for ind in all_gen_dict[gen]:
                all_fit.append(all_gen_dict[gen][ind][0])
                all_stiff.append(all_gen_dict[gen][ind][4])
                all_nclusters.append(clustering_allX_results[ind]['n_clusters'])
                all_Nclusters_voxels.append(clustering_allX_results[ind]['n_clusters']/np.sum(np.sum(np.sum(all_gen_dict[gen][ind][1] > 0))))

    return all_fit, all_stiff, all_nclusters,all_Nclusters_voxels



def create_modules_data(ENV,SIZE,SEED_INIT,SEED_END, EXP_NAME,MAX_GEN,CLUSTERING_NAME):
    all_fit, all_stiff, all_nclusters, all_env_names, all_sizes, all_strates = [],[],[],[],[], []
    all_nclustersRatio = []
    all_fit_this_exp = []
    d = {}
    for seed in range(SEED_INIT, SEED_END + 1):
            fit, stiff, nclusters, nclustersRatio = return_fit_stiff_nclusters(seed,EXP_NAME,MAX_GEN,CLUSTERING_NAME)
            all_fit_this_exp.extend(fit)
            all_stiff.extend(stiff)
            all_nclusters.extend(nclusters)
            all_nclustersRatio.extend(nclustersRatio)
            all_env_names.extend([ENV]*len(nclusters))
            all_sizes.extend([SIZE]*len(fit))
    max_fit = np.max(all_fit_this_exp)
    for fit in all_fit_this_exp:
            strate = round(fit/max_fit,1)
            strate = round(fit/max_fit,1)
            if strate >= 0.8:
                    all_strates.append('80-100%')
            elif strate >= 0.6:
                    all_strates.append('60-80%')
            elif strate >= 0.4:
                    all_strates.append('40-60%')
            elif strate >= 0.2:
                    all_strates.append('20-40%')
            else:
                    all_strates.append('0-20%')

    all_fit.extend(all_fit_this_exp)

    d = {'Nclusters': all_nclusters, 'Nclusters/Voxels':all_nclustersRatio,'fit': all_fit, 'stiff': all_stiff, 'Env':all_env_names,
    'Size':all_sizes, 'FitStrate': all_strates}
    
    with open("~/locomotion_principles/article_figs/Fig2/NumberModulesData/DictSegmentCalcs_{1}.pickle".format(EXP_NAME), 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return d


def create_avgdegree_data(ENV,SIZE,SEED_INIT,SEED_END, EXP_NAME,MAX_GEN,CLUSTERING_NAME):
    all_fit,all_stiffs,all_avg_deg,all_edges_nodes, all_Nedges,all_density,all_node_connectivity,all_diameter = [],[],[],[],[],[],[],[]
    all_strates= []
    all_exp_names = []
    all_fit_this_exp = []
    for seed in range(SEED_INIT, SEED_END + 1):
            print(seed)
            fit,stiffs,Nedges,density,node_connectivity,diameter, edges_nodes, avg_deg = TopologyCalcs(seed, EXP_NAME,MAX_GEN,CLUSTERING_NAME,'latin1')

            all_fit_this_exp.extend(fit)
            all_stiffs.extend(stiffs)
            all_edges_nodes.extend(edges_nodes)
            all_avg_deg.extend(avg_deg)
            all_Nedges.extend(Nedges)
            all_density.extend(density)
            all_node_connectivity.extend(node_connectivity)
            all_diameter.extend(diameter)
            all_exp_names.extend([ENV]*len(fit))

    max_fit = np.max(all_fit_this_exp)
    for fit in all_fit_this_exp:
            strate = round(fit/max_fit,1)
            if strate >= 0.8:
                    all_strates.append('80-100%')
            elif strate >= 0.6:
                    all_strates.append('60-80%')
            elif strate >= 0.4:
                    all_strates.append('40-60%')
            elif strate >= 0.2:
                    all_strates.append('20-40%')
            else:
                    all_strates.append('0-20%')

    all_fit.extend(all_fit_this_exp)
    

    d = {'Fitness': all_fit, 
    'Stiffness': all_stiffs,
    'Diameter': all_diameter,
    'NodeConnectivity': all_node_connectivity,
    'Nedges': all_Nedges,
    'Density': all_density,
    'AvgDegre':all_avg_deg, #2edges/nodes
    'Edges/Nodes':all_avg_deg, #2edges/nodes
    'FitStrate':all_strates,
    'Env':all_exp_names
    }
    
    with open("~/locomotion_principles/article_figs/Fig2/AvgDegreeData/TopologyCalcs_{1}.pickle".format(EXP_NAME), 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return d