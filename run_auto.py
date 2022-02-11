
##################### FUNCTIONS & MODDELS ###########################

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
from model import *

################## AUTO SCRIPT ####################


print('loading input data')
data = input_data()
data.build_input()

print('populating model')
md = model(data)

runs = [
        {'it_parameters.p_friction': 500, 'it_fields.vel_mes': data.vel_Millan, 'it_parameters.pmax':3000, 'it_parameters.delta_surf': 0.05},
#        {'it_parameters.p_friction': 1000, 'it_fields.vel_mes': data.vel_Jack, 'it_parameters.pmax':4995}
        ]

        


for element in runs:
    if isinstance(element, dict) == False:
        raise ValueError('{} not in correct format'.format(element))
        
save_strings = []
for q,run in enumerate(runs):
    save_string = ''
    for string in run:
        if isinstance(run[string], int):
            save_string += string + '_' + '{}'.format(run[string]) + '_'
        elif isinstance(run[string], np.ndarray):
            save_string += string + '_'
        elif isinstance(run[string], float):
            save_string += string + '_' + '{}'.format(round(run[string], 2)) + '_'
    if os.path.exists('./auto_ncs/Kronebreen_auto_1420_{}_{}.nc'.format(save_string, q)) or os.path.exists('./auto_data/Kronebreen_auto_1420_{}_{}.pkl'.format(save_string, q)):
        raise ValueError('run Kronebreen_auto_1420_{}_{} already exists'.format(save_string, q))
    save_strings.append(save_string)

for i,run in enumerate(runs):
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
        with open('./auto_data/Kronebreen_auto_1420_{}_{}.pkl'.format(save_strings[i], i), 'wb') as outp:
            pickle.dump(md, outp, pickle.HIGHEST_PROTOCOL)
        subprocess.call(['cp', md.file_locations.it_out, './auto_ncs/Kronebreen_auto_1420_{}_{}.nc'.format(save_strings[i], i)])
        print('saving done, this was run {} out of {} runs'.format(i+1, len(runs)))
        md.__init__(data)
    except:
        print('something went wrong in iteration or during saving')
        md.warnings.append('something went wrong in iteration or during saving')
        md.__init__(data)
        continueGGG
