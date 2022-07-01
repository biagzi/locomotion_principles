import os
import sys
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pylab as plt

PC_NAME = 'renatabb'

FROM_ENV = str(sys.argv[1])
TO_ENV = str(sys.argv[2])
SELECT_ROBOT = str(sys.argv[3]) #BEST or WORST
RUN_DIR = str(sys.argv[4])

sys.path.insert(0, os.path.abspath('/home/{0}/locomotion_principles'.format(PC_NAME)))
from softbot import Genotype, Phenotype
MyPhenotype = Phenotype
MyGenotype = Genotype


sys.path.insert(0, os.path.abspath('/home/{0}/locomotion_principles/data_analysis/TransferenceSelectedShapes/').format(PC_NAME))
from cluster_TransferenceSelectedShape import ClusterizationSimulationSelection_MainAlgorithm


SIZE = 4
EPS_LIST = [1.,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
EPS_PO_LIST = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
min_samples = 2
COARSE_GRAIN = True




DoClusterAndSim = True #If False, it will just do Selection and will not check for a eps/eps_po that was not done yet.

if TO_ENV == 'ALL':
    for TO_ENV in ['AQUA','MARS','EARTH']:
        ClusterizationSimulationSelection_MainAlgorithm(FROM_ENV, TO_ENV,SIZE,EPS_LIST, EPS_PO_LIST,min_samples,COARSE_GRAIN,DoClusterAndSim=DoClusterAndSim,SELECT_ROBOT=SELECT_ROBOT,RUN_DIR = RUN_DIR)
else:
    ClusterizationSimulationSelection_MainAlgorithm(FROM_ENV, TO_ENV,SIZE,EPS_LIST, EPS_PO_LIST,min_samples,COARSE_GRAIN,DoClusterAndSim=DoClusterAndSim,SELECT_ROBOT=SELECT_ROBOT,RUN_DIR = RUN_DIR)



#Command to run in terminal:
##nice -n 19 nohup sh -c " PYTHONPATH=$HOME/locomotion_principles/ $HOME/miniconda2/bin/python $HOME/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/RunDoClusterization.py AQUA AQUA " > /dev/null &

##nice -n 19  nohup  sh -c " PYTHONPATH=$HOME/locomotion_principles/ $HOME/miniconda2/bin/python $HOME/locomotion_principles/data_analysis/ClusteredTransferenceSelectedShapes/RunDoClusterization.py MARS AQUA BEST CPPN_Inovation" > /dev/null &

#RUN_DIR option: "CPPN_Inovation","DE_CPPN_Inovation", "DE_Inovation","CPPN", "DE_CPPN", "DE", 
