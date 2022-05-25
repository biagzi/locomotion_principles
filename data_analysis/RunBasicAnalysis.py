import matplotlib
matplotlib.use('Agg')
import sys
import os

#Use python 2

from softbot import Genotype, Phenotype
MyPhenotype = Phenotype
MyGenotype = Genotype


DO_EXPS = []

for args_indx in range(1,len(sys.argv)):
    DO_EXPS.append(str(sys.argv[args_indx]))

MORE = False

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


    try:
        os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}".format(EXP_NAME))
        os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts".format(EXP_NAME))
    except OSError:
        print ("Creation of the directory failed. Check if the directory already exist.")
    else:
        print ("Successfully created the directory ")

    ##################################################################################################################################################################
    ################ Generate dicts for each seed with just the most important information of all different softbot created in evolution ###############################
    
    DO = True

    if DO is True:
        from data_analysis.BasicAnalysisUtils import generate_dicts_allrobots_per_seed_with_stiff
        print("START - generating dicst")
        generate_dicts_allrobots_per_seed_with_stiff(EXP_NAME,EXP_NAME,MAX_GEN,SEED_INIT,SEED_END)
        print('FINISH - all robots dict of {0}'.format(EXP_NAME))


