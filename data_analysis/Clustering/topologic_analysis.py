import numpy as np
import pandas as pd
import pickle
import networkx as nx
from basic_analysis_utils import return_fit_stiff

def TopologyCalcs(seed, EXP_NAME,MAX_GEN,CLUSTERING_NAME,encode):

    all_fits = []
    all_stiffs = []
    all_Nedges = []
    all_density = []
    all_node_connectivity = []
    all_diameter = []
    all_avg_deg = []
    all_avg_deg_Calculated = []
    
    ProcessedClusterInfos_location = "exp_analysis/{0}/clustering/ProcessedClusterInfos".format(EXP_NAME)
    with open("/~/locomotion_principles/data_analysis/{0}/ProcessedClusters_{1}_seed{2}.pickle".format(ProcessedClusterInfos_location,CLUSTERING_NAME,seed), 'rb') as handle:
        if encode== "ASCII":
            clusters_processed = pickle.load(handle)
        else:
            clusters_processed = pickle.load(handle,encoding=encode)

    all_ind_fit, all_ind_stiff = return_fit_stiff(seed, EXP_NAME,MAX_GEN)

    
    for ind_id in all_ind_fit:

        dfnumpy = clusters_processed[ind_id]['connection'].to_numpy(np.dtype)
        dfnumpy[dfnumpy== True] = 1
        dfnumpy[dfnumpy== None] = 0
        dfnumpy[dfnumpy== False] = 0
        dfnumpy = np.array(dfnumpy,dtype=int)
        G = nx.from_numpy_array(dfnumpy) 

        if nx.is_connected(G):
            all_node_connectivity.append(nx.algorithms.connectivity.connectivity.node_connectivity(G)) #Node connectivity is equal to the minimum number of nodes that must be removed to disconnect G or render it trivial.
            all_fits.append(all_ind_fit[ind_id])
            all_diameter.append(nx.algorithms.distance_measures.diameter(G)) #The diameter is the maximum eccentricity: The eccentricity of a node v is the maximum distance from v to all other nodes in G.
            all_Nedges.append(G.number_of_edges())
            all_density.append(nx.density(G)) #d = 2m/(n(n-1)) where n is the number of nodes and m is the number of edges in G.
            #The density is 0 for a graph without edges and 1 for a complete graph. The density of multigraphs can be higher than 1.)
            all_avg_deg.append(float(G.number_of_edges())/G.number_of_nodes())
            all_avg_deg_Calculated.append(2*float(G.number_of_edges())/G.number_of_nodes())
            all_stiffs.append(all_ind_stiff[ind_id])

    return all_fits,all_stiffs,all_Nedges,all_density,all_node_connectivity,all_diameter,all_avg_deg,all_avg_deg_Calculated