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

def return_fit_shapesym_posym(seed, TITLE_NAME,MAX_GEN,COMPUTER_NAME,SIZE, encode = "ASCII"):
    """
    Return shape symmetry and phase offset symmetry values of the robots
    """
    #encode = "latin1" # unpickling a python 2 object with python 3

    all_fits = []
    all_shapeXsym = []
    all_shapeYsym = []
    all_shapeSYM = []
    all_poSYM = []
    all_poXsym = []
    all_poYsym = []
    all_shape_majorsymmetry,all_shape_minussymmetry = [],[]
    all_phase_majorsymmetry,all_phase_minussymmetry = [],[]
    all_stiff = []

    if os.path.isfile("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(TITLE_NAME,seed,COMPUTER_NAME)) is True:
        with open("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(TITLE_NAME,seed,COMPUTER_NAME), 'rb') as handle:
            if encode == "ASCII":
                all_gen_dict = pickle.load(handle)
            else:
                all_gen_dict = pickle.load(handle,encoding=encode)
    else:
        print ('do not found seed')
        print ("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts/seed_{1}.pickle".format(TITLE_NAME,seed,COMPUTER_NAME))
        return
    
    for gen in all_gen_dict:
        if gen <= MAX_GEN:
            for ind in all_gen_dict[gen]:
                fit, shape, po = all_gen_dict[gen][ind][0], np.abs(all_gen_dict[gen][ind][1]/9), all_gen_dict[gen][ind][2]
                stiff = all_gen_dict[gen][ind][4]
                
                for x in range(SIZE):
                    for y in range(SIZE):
                        for z in range (SIZE):
                            if shape[x,y,z] == 0:
                                po[x,y,z] = 0

                NVoxels = float(np.sum(shape))

                shape_Xsym = 1 - (np.count_nonzero(shape - np.flip(shape,0))/2.)/NVoxels
                shape_Ysym = 1 - (np.count_nonzero(shape - np.flip(shape,1))/2.)/NVoxels

                po_Xsym = 1 - np.count_nonzero(np.around(po - np.flip(po,0),2))/2./NVoxels
                po_Ysym = 1 - np.count_nonzero(np.around(po - np.flip(po,1),2))/2./NVoxels

                all_fits.append(fit)
                all_stiff.append(stiff)
                all_shapeXsym.append(shape_Xsym)
                all_shapeYsym.append(shape_Ysym)
                all_shapeSYM.append((shape_Xsym + shape_Ysym)/2)
                all_poXsym.append(po_Xsym)
                all_poYsym.append(po_Ysym)
                all_poSYM.append((po_Xsym+po_Ysym)/2)
                if shape_Xsym >= shape_Ysym:
                    all_shape_majorsymmetry.append(shape_Xsym)
                    all_shape_minussymmetry.append(shape_Ysym)
                else:
                    all_shape_majorsymmetry.append(shape_Ysym)
                    all_shape_minussymmetry.append(shape_Xsym)
                if po_Xsym >= po_Ysym:
                    all_phase_majorsymmetry.append(po_Xsym)
                    all_phase_minussymmetry.append(po_Ysym)
                else:
                    all_phase_majorsymmetry.append(po_Ysym)
                    all_phase_minussymmetry.append(po_Xsym)
                
                # all_fits_symmetry[ind] = {'fit':fit,'shape_Xsym':shape_Xsym,'shape_Ysym':shape_Ysym,
                #                             'po_Xsym':po_Xsym,'po_Ysym':po_Ysym}
        
    return all_fits,all_stiff, all_shapeXsym,all_shapeYsym,all_poXsym, all_poYsym,all_shapeSYM,all_poSYM,all_shape_majorsymmetry,all_shape_minussymmetry,all_phase_majorsymmetry,all_phase_minussymmetry




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


def create_symmetry_data(ENV,SIZE,SEED_INIT,SEED_END, EXP_NAME,MAX_GEN):
        all_fit, all_stiffs,  all_exp_names, all_strates,all_shapeXsym,all_shapeYsym,all_poXsym, all_poYsym,all_shapeSYM,all_poSYM = [],[],[],[],[],[],[],[],[],[]
        all_shapeMajor,all_shapeMinor,all_phaseMajor,all_phaseMinor = [],[],[],[]
        all_fit_this_exp = []
        for seed in range(SEED_INIT, SEED_END + 1):
                fits,stiffs, shapeXsym,shapeYsym,poXsym, poYsym, shapeSYM,poSYM,shapeMajor,shapeMinor, poMajor,poMinor = return_fit_shapesym_posym(seed, EXP_NAME,MAX_GEN,COMPUTER_NAME,SIZE,'ASCII')
                all_shapeMajor.extend(shapeMajor)
                all_shapeMinor.extend(shapeMinor)
                all_phaseMajor.extend(poMajor)
                all_phaseMinor.extend(poMinor)
                all_fit_this_exp.extend(fits)
                all_stiffs.extend(stiffs)
                all_shapeXsym.extend(shapeXsym)
                all_shapeYsym.extend(shapeYsym)
                all_poXsym.extend(poXsym)
                all_poYsym.extend(poYsym)
                all_shapeSYM.extend(shapeSYM)
                all_poSYM.extend(poSYM)
                all_exp_names.extend([ENV]*len(fits))

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

        d = {   'Fitness': all_fit, 'Exp':all_exp_names, 'Stiff':all_stiffs,
        'XShapeSymmetry':all_shapeXsym,'YShapeSymmetry':all_shapeYsym,
        'XYShapeSymmetry':all_shapeSYM,
        'XPhaseSymmetry':all_poXsym,'YPhaseSymmetry':all_poYsym,
        'XYPhaseSymmetry':all_poSYM,
        'ShapeMajorSymmetry':all_shapeMajor,'ShapeMinorSymmetry':all_shapeMinor,
        'PhaseMajorSymmetry':all_phaseMajor,'PhaseMinorSymmetry':all_phaseMinor,
        'FitStrate': all_strates}

        with open("~/locomotion_principles/article_figs/Fig2/AvgDegreeData/DictSymmetryCalcs_{1}.pickle".format(EXP_NAME), 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return d