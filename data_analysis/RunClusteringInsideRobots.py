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

        from Clustering.clustering_algorithm import algorithm_cluster_each_seed

        if os.path.isdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/clustering/seeds_clustered_{1}".format(EXP_NAME,CLUSTERING_NAME)) is False:
            os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/clustering/seeds_clustered_{1}".format(EXP_NAME,CLUSTERING_NAME))

        print('START Clustering')
        algorithm_cluster_each_seed(SEED_INIT, SEED_END,EPS,min_samples, EXP_NAME, SIZE,MAX_GEN,CLUSTERING_NAME,eps_po=EPS_PO,COARSE_GRAIN = COARSE_GRAIN,factor = FACTOR,encode = ENCODE)
        print('FINISH Clustering')

##################################################################################################################################################################
############################## CREATE TOPOLOGIC REPRESENTATION BY PROCESSING EACH CLUSTER SEED  #########################################################################################################
    DO = True
    if DO is True:
        if os.path.isdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/clustering/ProcessedClusterInfos".format(EXP_NAME)) is False:
            os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/clustering/ProcessedClusterInfos".format(EXP_NAME))
        
        from Clustering.topologic_representation import process_clusters

        print('START Process clustering')
        process_clusters(SEED_INIT,SEED_END,EXP_NAME,CLUSTERING_NAME,ENCODE,MAX_GEN,SIZE)
        print('FINISH Process clustering')


##################################################################################################################################################################
############################################  SNAPSHOT OF CLUSTERED ROBOTS #########################################################################################################
    DO = True
    if DO is True:

        from Clustering.clustering_vizualization import print_cluster_po_by_side
        
        SEED = 1
        SAVE_FIG = True
        NUMBER_OF_FIGS = 30
        for I_VALUE in [1000,2000]:
            print_cluster_po_by_side(SEED,I_VALUE, NUMBER_OF_FIGS,SIZE,EXP_NAME,MAX_GEN,CLUSTERING_NAME,SAVE_FIG,encode=ENCODE)
            print('FINISH PLOT SNAPSHOT CLUSTERED i value = {0}'.format(I_VALUE))




