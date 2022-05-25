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
    if exp == '4.a':
        EXP_NAME = 'Final_Lp4_AQUA'
        TITLE_NAME = 'Final_4_AQUA'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.e':
        SEED_INIT = 1
        SEED_END = 30
        EXP_NAME = 'Final_Lp4_EARTH'
        TITLE_NAME = 'Final_4_EARTH'
        SIZE = [4,4,4]
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        COLOR = 'darkseagreen'
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.m':
        EXP_NAME = 'Final_Lp4_MARS'
        TITLE_NAME = 'Final_4_MARS'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '4.a.dc':
        EXP_NAME = 'Final_Lp4_AQUA_DirectEncode'
        TITLE_NAME = 'Final_4_AQUA_DirectEncode'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 20
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.m.dc':
        EXP_NAME = 'Final_Lp4_MARS_DirectEncode'
        TITLE_NAME = 'Final_4_MARS_DirectEncode'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 20
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '4.e.dc':
        EXP_NAME = 'Final_Lp4_EARTH_DirectEncode'
        TITLE_NAME = 'Final_4_EARTH_DirectEncode'
        SIZE = [4,4,4]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 20
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.a':
        EXP_NAME = 'Final_Lp6_AQUA'
        TITLE_NAME = 'Final_6_AQUA'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.e':
        EXP_NAME = 'Final_Lp6_EARTH'
        TITLE_NAME = 'Final_6_EARTH'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.m':
        EXP_NAME = 'Final_Lp6_MARS'
        TITLE_NAME = 'Final_6_MARS'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 30
        print ('entered in {0}'.format(EXP_NAME))


    if exp == '6.a.dc':
        EXP_NAME = 'Final_Lp6_AQUA_DirectEncode'
        TITLE_NAME = 'Final_6_AQUA_DirectEncode'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 10
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.m.dc':
        EXP_NAME = 'Final_Lp6_MARS_DirectEncode'
        TITLE_NAME = 'Final_6_MARS_DirectEncode'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 10
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '6.e.dc':
        EXP_NAME = 'Final_Lp6_EARTH_DirectEncode'
        TITLE_NAME = 'Final_6_EARTH_DirectEncode'
        SIZE = [6,6,6]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 10
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))


    if exp == '8.a':
        EXP_NAME = 'Final_Lp8_AQUA'
        TITLE_NAME = 'Final_8_AQUA'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '8.e':
        EXP_NAME = 'Final_Lp8_EARTH'
        TITLE_NAME = 'Final_8_EARTH'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '8.m':
        EXP_NAME = 'Final_Lp8_MARS'
        TITLE_NAME = 'Final_8_MARS'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '8.a.de':
        EXP_NAME = 'Final_Lp8_AQUA_DirectEncode'
        TITLE_NAME = 'Final_8_AQUA_DirectEncode'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 5
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))

    if exp == '8.e.de':
        EXP_NAME = 'Final_Lp8_EARTH_DirectEncode'
        TITLE_NAME = 'Final_8_EARTH_DirectEncode'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 2
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))
    
    if exp == '8.m.de':
        EXP_NAME = 'Final_Lp8_MARS_DirectEncode'
        TITLE_NAME = 'Final_8_MARS_DirectEncode'
        SIZE = [8,8,8]
        COLOR = 'darkseagreen'
        MAX_GEN = 1500
        SEED_INIT = 1
        SEED_END = 2
        MORE = True
        print ('entered in {0}'.format(EXP_NAME))


    try:
        os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}".format(TITLE_NAME))
        os.mkdir("~/locomotion_principles/data_analysis/exp_analysis/{0}/seeds_dicts".format(TITLE_NAME))
    except OSError:
        print ("Creation of the directory failed. Check if the directory already exist.")
    else:
        print ("Successfully created the directory ")

    ##################################################################################################################################################################
    ################ Generate dicts for each seed with just the most important information of all different softbot created in evolution ###############################
    
    DO = True

    if DO is True:
        from AnalysisUtils import generate_dicts_allrobots_per_seed_with_stiff
        print("START - generating dicst")
        generate_dicts_allrobots_per_seed_with_stiff(EXP_NAME,TITLE_NAME,MAX_GEN,SEED_INIT,SEED_END)
        print('FINISH - all robots dict of {0}'.format(EXP_NAME))


