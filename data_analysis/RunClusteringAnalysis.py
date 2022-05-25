import sys
import os

DO_EXPS = []

for args_indx in range(1,len(sys.argv)):
    DO_EXPS.append(str(sys.argv[args_indx]))

NORMALIZED = 3.2

#PYTHON 3
ENCODE = 'latin1'
print('Using python {0}'.format(sys.version_info[0]))

for exp in DO_EXPS:
    if exp == '4.w':
        EXP_NAME = '4_Water_CPPN'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.e':
        SEED_INIT = 1
        SEED_END = 30
        EXP_NAME = '4_Earth_CPPN'
        SIZE = [4,4,4]
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        COLOR = 'darkseagreen'
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.m':
        EXP_NAME = '4_Mars_CPPN'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '4.w.de':
        EXP_NAME = '4_Water_DE'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 20
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.m.de':
        EXP_NAME = '4_Mars_DE'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 20
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.e.de':
        EXP_NAME = '4_Earth_DE'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 20
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.w':
        EXP_NAME = '6_Water_CPPN'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.e':
        EXP_NAME = '6_Earth_CPPN'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.m':
        EXP_NAME = '6_Mars_CPPN'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))


    if exp == '6.w.de':
        EXP_NAME = '6_Water_DE'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 10
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.m.de':
        EXP_NAME = '6_Mars_DE'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 10
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.e.de':
        EXP_NAME = '6_Earth_DE'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 10
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))


    if exp == '8.w':
        EXP_NAME = '8_Water_CPPN'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '8.e':
        EXP_NAME = '8_Earth_CPPN'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '8.m':
        EXP_NAME = '8_Mars_CPPN'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '8.w.de':
        EXP_NAME = '8_Water_DE'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '8.e.de':
        EXP_NAME = '8_Earth_DE'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 2
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '8.m.de':
        EXP_NAME = '8_Mars_DE'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 2
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    #######################

    EPS = 1.2
    EPS_PO = 0.02
    min_samples = 2

    FACTOR = 10
    COARSE_GRAIN = True

    CLUSTERING_NAME = "{0}-{1}".format(EPS,EPS_PO)
    CLUSTERING_FOLDER_NAME = 'clustering'


##################################################################################################################################################################
############################################  CLUSTER EACH SEED AND SAVE RESULTS IN DICT #########################################################################################################
    DO = True
    if DO is True:

        #Clustering 
        if os.path.isdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/clustering".format(EXP_NAME)) is False:
            os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/clustering".format(EXP_NAME))

        from save_open_clustering import algorithm_cluster_each_seed

        if os.path.isdir("/home/{1}/reconfigurable_organisms/data_analysis/exp_analysis/{0}/clustering/seeds_clustered_{2}".format(EXP_NAME,COMPUTER_NAME,CLUSTERING_NAME)) is False:
            os.mkdir("/home/{1}/reconfigurable_organisms/data_analysis/exp_analysis/{0}/clustering/seeds_clustered_{2}".format(EXP_NAME,COMPUTER_NAME,CLUSTERING_NAME))

        print('START Clustering')
        algorithm_cluster_each_seed(SEED_INIT, SEED_END,EPS,min_samples, EXP_NAME, SIZE,MAX_GEN,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,eps_po=EPS_PO,COARSE_GRAIN = COARSE_GRAIN,factor = FACTOR,encode = ENCODE)
        print('FINISH Clustering')

##################################################################################################################################################################
############################################  SNAPSHOT OF CLUSTERED ROBOTS #########################################################################################################
    DO = True
    if DO is True:
        sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
        from clustering_plots_from_seeds import print_cluster_po_by_side
        
        SEED = 1
        SAVE_FIG = True
        NUMBER_OF_FIGS = 30
        for I_VALUE in [1000,2000]:
            print_cluster_po_by_side(SEED,I_VALUE, NUMBER_OF_FIGS,SIZE,EXP_NAME,MAX_GEN,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,SAVE_FIG,encode=ENCODE)
            print('FINISH PLOT SNAPSHOT CLUSTERED i value = {0}'.format(I_VALUE))

##################################################################################################################################################################
############################################  Plot of fitXstiff and fitXnclusters #########################################################################################################
    DO = True

    if DO is True:
        sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
        from clustering_plots_from_seeds import plot_fit_stiff_nclusters

        plot_fit_stiff_nclusters(SEED_INIT,SEED_END,EXP_NAME,MAX_GEN,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,encode = ENCODE)
        print("FINISH - fitXstiff and fitXnclusters plots")

##################################################################################################################################################################
############################################  Plot PoAmplitudeXnclusters #########################################################################################################
    DO = True

    if DO is True:
        sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
        from clustering_plots_from_seeds import collect_plot_distribution_po_amplitude_per_cluster

        collect_plot_distribution_po_amplitude_per_cluster(SEED_INIT,SEED_END,MAX_GEN,EXP_NAME,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,ENCODE,CLUSTERING_FOLDER_NAME,SIZE)
        print("FINISH - PoAmplitudeXnclusters plots")


##################################################################################################################################################################
############################################  PLOTS USING BEST OF #########################################################################################################
    DO = True

    if DO is True:
        sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
        from clustering_plots_from_seeds import collect_best_of_each_number_of_clusters_and_stiff,plot_best_of_stiff_best_of_cluster,dist_best_of_cluster

        best_of_stiffs, best_of_nclusters = collect_best_of_each_number_of_clusters_and_stiff(SEED_INIT,SEED_END,MAX_GEN,EXP_NAME,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,ENCODE,SIZE)

        #####  Plot best of each number of clusters ##############
        plot_best_of_stiff_best_of_cluster(best_of_stiffs, best_of_nclusters,SEED_INIT,SEED_END,EXP_NAME,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,SIZE)
        print("FINISH - BEST OF STIFF AND NCLUS")

        #####  Plot best of each number of clusters DIST ##############
        dist_best_of_cluster(best_of_nclusters,EXP_NAME,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,SIZE)
        print("FINISH - DIST BEST OF NCLUS")

    
##################################################################################################################################################################
############################################  PLOTS CLUSTER EVOLUTION #########################################################################################################
    DO = True

    if DO is True:
        sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
        from clustering_plots_from_seeds import cluster_evolution, ploster_cluster_evolution

        #####  Plot best of each number of clusters ##############
        cluster_evolution(SEED_INIT,SEED_END,EXP_NAME,EXP_NAME,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,ENCODE)
        print("FINISH - CREATE PICKLE CLUSTER EVOLUTION")

        #####  Plot best of each number of clusters DIST ##############
        ploster_cluster_evolution(EXP_NAME)
        print("FINISH -PLOT CLUSTER EVOLUTION")


##################################################################################################################################################################
############################################  COMPARE ENVS #########################################################################################################
EPS = 1.2
EPS_PO = 0.02
min_samples = 2

FACTOR = 10
COARSE_GRAIN = True

CLUSTERING_NAME = "{0}-{1}".format(EPS,EPS_PO)
CLUSTERING_FOLDER_NAME = 'clustering'


DO = False
if DO is True:
    sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
    from clustering_plots_from_seeds import plot_fit_nclusters_all_envs_FINAL
    
    all_exps_infos = {'Final_4_AQUA':[1,30,4],'Final_4_EARTH':[1,30,4],'Final_4_MARS':[1,30,4]}
    add_name = '_CPPN'

    # all_exps_infos = {'Final_4_AQUA_DirectEncode':[1,10,4],'Final_4_EARTH_DirectEncode':[1,10,4],
    #                         'Final_4_MARS_DirectEncode':[1,10,4]}
    # add_name = '_DirectEncode'

    #FINAL IS FOR PRESENTATION 
    plot_fit_nclusters_all_envs_FINAL(all_exps_infos,1500,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,encode = ENCODE,add_name=add_name)
    print("FINISH - fitXnclusters all exps")


# DO = False
# if DO is True:
#     sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
#     from clustering_plots_from_seeds import plot_fit_nclusters_all_envs
    
#     plot_fit_nclusters_all_envs(all_exps_infos,1500,COMPUTER_NAME,FOLDER_LOCATION,CLUSTERING_NAME,encode = ENCODE)
#     print("FINISH - fitXnclusters all exps")

##################################################################################################################################################################
############################################  COMPARE ENVS AND PARAMS #########################################################################################################


DO = False
if DO is True:
    sys.path.insert(0, os.path.abspath('~/locomotion_principles/data_analysis/Clustering'.format(COMPUTER_NAME))) 
    from clustering_plots_from_seeds import plot_fit_nclusters_all_envs_compare_params, plot_distribution_FitNclusters

    # all_params = [[1.1,0.03],
    #             [1.4,0.03],[1.6,0.03],[1.8,0.03]
    #                 ]
    all_params = [[1.6,0.03]]
    add_name = '_CPPNxDE'

    col_wrap = 3
    all_exps_infos = {'Final_4_AQUA':[1,10,4],'Final_4_EARTH':[1,10,4],'Final_4_MARS':[1,10,4],
                    'Final_4_AQUA_DirectEncode':[1,10,4],'Final_4_EARTH_DirectEncode':[1,10,4],'Final_4_MARS_DirectEncode':[1,10,4],
                        }
    
    plot_fit_nclusters_all_envs_compare_params(all_params,all_exps_infos,1500,COMPUTER_NAME,FOLDER_LOCATION,encode = ENCODE,add_name = add_name,col_wrap = col_wrap)
    print("FINISH - fitXnclusters all exps")

    #plot_distribution_FitNclusters(all_params,all_exps_infos,1500,COMPUTER_NAME,FOLDER_LOCATION,encode = ENCODE,add_name = add_name)
    print("FINISH - fit dISTRIBUTION per nclusters all exps")


