import sys
import getopt
import math
import numpy as np
from netCDF4 import Dataset as NC
import matplotlib.pyplot as plt
import tqdm
from scipy import ndimage
from scipy import spatial
from copy import deepcopy
import subprocess
import rasterio
from rasterio.plot import show
import geopandas as gpd
from funcs import *
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error
from rasterio import mask as msk
import pandas as pd
import scipy.io
import pickle
import time
import multiprocessing
import richdem as rd
from matplotlib import colors as plt_colors

from model_icecap import *

################## AUTO SCRIPT ####################


print('loading input data')
data = input_data()
data.build_input()

print('populating model')
md = model(data)

xt, yt = np.ogrid[:np.shape(md.it_fields.S_ref)[0], :np.shape(md.it_fields.S_ref)[1]]
np.random.seed(0)

runs = [
    ####### INPUT DATA ERRORS #######
    ### smb errors ###
    {'it_fields.smb': md.it_fields.smb * 1.1},# 'it_fields.dh_ref': md.it_fields.smb * 0.1 / md.it_parameters.ice_density},
    {'it_fields.smb': md.it_fields.smb * 1.3},# 'it_fields.dh_ref': md.it_fields.smb * 0.3 / md.it_parameters.ice_density},
    {'it_fields.smb': md.it_fields.smb * 1.5},# 'it_fields.dh_ref': md.it_fields.smb * 0.5 / md.it_parameters.ice_density},
    {'it_fields.smb': md.it_fields.smb * 0.9},# 'it_fields.dh_ref': md.it_fields.smb * -0.1 / md.it_parameters.ice_density},
    {'it_fields.smb': md.it_fields.smb * 0.7},# 'it_fields.dh_ref': md.it_fields.smb * -0.3 / md.it_parameters.ice_density},
    {'it_fields.smb': md.it_fields.smb * 0.5},# 'it_fields.dh_ref': md.it_fields.smb * -0.5 / md.it_parameters.ice_density},
    {'it_fields.smb': md.it_fields.smb * np.random.normal(1, 1, np.shape(md.it_fields.S_ref))},
    {'it_fields.smb': md.it_fields.smb * np.random.normal(1, 2, np.shape(md.it_fields.S_ref))},

    ### dhdt errors ### --> covered above

    ### S_ref errors ###
    {'it_fields.S_ref': np.maximum(md.it_fields.S_ref * np.random.normal(1, 0.02, np.shape(md.it_fields.S_ref)), md.it_fields.B_rec)},
    {'it_fields.S_ref': np.maximum(md.it_fields.S_ref * np.random.normal(1, 0.04, np.shape(md.it_fields.S_ref)), md.it_fields.B_rec)},
    {'it_fields.S_ref': md.it_fields.S_ref + np.maximum(-((yt*2-30) ** 2 + (xt*2-30) ** 2)+50, 0)}, #bump of 50 m height
    {'it_fields.S_ref': md.it_fields.S_ref + np.maximum(-((yt*3-47) ** 2 + (xt*3-47) ** 2)+100, 0)}, #bump of 100 m height

    ### velocity errors ###
    {'it_fields.vel_mes': np.maximum(md.it_fields.vel_mes * np.random.normal(1, 0.2, np.shape(md.it_fields.S_ref)), 0)},
    {'it_fields.vel_mes': np.maximum(md.it_fields.vel_mes * np.random.normal(1, 0.5, np.shape(md.it_fields.S_ref)), 0)},
    {'it_fields.vel_mes': md.it_fields.vel_mes * 1.1},
    {'it_fields.vel_mes': md.it_fields.vel_mes * 1.3},
    {'it_fields.vel_mes': md.it_fields.vel_mes * 1.5},
    {'it_fields.vel_mes': md.it_fields.vel_mes * 0.9},
    {'it_fields.vel_mes': md.it_fields.vel_mes * 0.7},
    {'it_fields.vel_mes': md.it_fields.vel_mes * 0.5},

    ### errors in mask ###
    {'it_fields.mask': np.ones_like(md.it_fields.S_ref)},
    
    ### errors in A ###
    {'it_parameters.A': 1.733e3*np.exp(-13.9e4/(8.3*264))},
    {'it_parameters.A': 1.733e3*np.exp(-13.9e4/(8.3*266))},
    {'it_parameters.A': 1.733e3*np.exp(-13.9e4/(8.3*270))},
    {'it_parameters.A': 1.733e3*np.exp(-13.9e4/(8.3*272))},
    
    ### errors in ice density ###
    {'it_parameters.ice_density': 850},
    {'it_parameters.ice_density': 900},
    {'it_parameters.ice_density': 1500},
    
    ###  --> leave for another day  <-- ###
    ####### TESTING DIFFERENT INVERSION SETTINGS #######
    ### delta surf ###

    ### beta ###

    ### dt ###

    ### bw ###

    ### shift ###

    ### p_friction ###

    ### tauc_scale ###

    ### initial bed guess ###

    ### initial tauc guess ###
        ]


for element in runs:
    if isinstance(element, dict) == False:
        raise ValueError('{} not in correct format'.format(element))
        
save_strings = []
q_skip = []
for q,run in enumerate(runs):
    save_string = ''
    for string in run:
        if isinstance(run[string], int):
            save_string += string + '_' + '{}'.format(run[string]) + '_'
        elif isinstance(run[string], np.ndarray):
            save_string += string + '_'
        elif isinstance(run[string], float):
            save_string += string + '_' + '{}'.format(round(run[string], 2)) + '_'
    if os.path.exists('./auto_ncs/icecap_auto_{}_{}.nc'.format(save_string, q)) or os.path.exists('./auto_data/icecap_auto_{}_{}.pkl'.format(save_string, q)):
        q_skip.append(q)
        print('run icecap_auto_{}_{} already exists; skipping'.format(save_string, q))
    save_strings.append(save_string)

for i,run in enumerate(runs):
    if i in q_skip:
        continue
    for key in run:
        try:
            key_split = key.split('.')
        except:
            print('key {} could not be split'.format(key))
            md.warnings.append('key {} could not be split'.format(key))
            continue
        try:
            md.__dict__[key_split[0]].__dict__[key_split[1]] = run[key]
        except:
            print('parameter {} could not be set'.format(key))
            md.warnings.append('parameter {} could not be set'.format(key))
            continue
    print('populating model {} done'.format(save_strings[i]))
   
    try:
        print('now iterating...')
        md.iterate(data)
        print('iterating done, now saving')
        with open('./auto_data/icecap_auto_{}_{}.pkl'.format(save_strings[i], i), 'wb') as outp:
            pickle.dump(md, outp, pickle.HIGHEST_PROTOCOL)
        subprocess.call(['cp', md.file_locations.it_out, './auto_ncs/icecap_auto_{}_{}.nc'.format(save_strings[i], i)])
        print('saving done, this was run {} out of {} runs'.format(i+1, len(runs)))
        md.__init__(data)
    except:
        print('something went wrong in iteration or during saving')
        md.warnings.append('something went wrong in iteration or during saving')
        md.__init__(data)
        continue
