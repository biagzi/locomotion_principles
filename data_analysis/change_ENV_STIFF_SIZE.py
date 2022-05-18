import pickle
from glob import glob
import subprocess as sub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from mpl_toolkits.mplot3d import Axes3D


sys.path.insert(0, os.path.abspath('~/locomotion_principles'))

from softbot import Genotype, Phenotype
from base import Env
from tools.read_write_voxelyze import write_voxelyze_file

MyPhenotype = Phenotype
MyGenotype = Genotype



def choose_my_env(new_env):
    if new_env == 'WATER':
        TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
        FREQ = 2
        GRAV_ACC = -0.1
        VOXEL_SIZE = 0.01  # meters

    elif new_env == 'EARTH':
        TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
        FREQ = 2
        GRAV_ACC = -9.81
        VOXEL_SIZE = 0.01  # meters

    elif new_env == 'MARS':
        TEMP_AMP = 39.4714242553  # 50% volumetric change with temp_base=25: (1+0.01*(39.4714242553-25))**3-1=0.5
        FREQ = 2
        GRAV_ACC = -3.721
        VOXEL_SIZE = 0.01  # meters
        
    my_env = Env(temp_amp=TEMP_AMP, frequency=FREQ,lattice_dimension=VOXEL_SIZE, grav_acc=GRAV_ACC)
    return my_env