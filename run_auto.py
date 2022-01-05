
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


                
class input_data():
    
    def __init__(self):
            self.x, self.y = np.meshgrid(1,1)
            self.smb = np.zeros_like(self.x, dtype='float')
            self.dem = np.zeros_like(self.x, dtype='float')
            self.dhdt = np.zeros_like(self.x, dtype='float')
            self.mask = np.zeros_like(self.x, dtype='float')
            self.vel = np.zeros_like(self.x, dtype='float')
            self.vx = np.zeros_like(self.x, dtype='float')
            self.vy = np.zeros_like(self.x, dtype='float')
    
    def check_shape(self):
        check_shape(self.x, [self.dem, self.smb, self.dhdt, self.mask, self.vel, self.vx, self.vy])
                    
    def reset_shape(self):
        self.dem = np.zeros_like(self.x, dtype='float')
        self.smb = np.zeros_like(self.x, dtype='float')
        self.dhdt = np.zeros_like(self.x, dtype='float')
        self.mask = np.zeros_like(self.x, dtype='float')
        self.vel = np.zeros_like(self.x, dtype='float')
        self.vx = np.zeros_like(self.x, dtype='float')
        self.vy = np.zeros_like(self.x, dtype='float')
                
    def set_xy(self, matrix):
        self.x, self.y = np.meshgrid(matrix[0],np.arange(np.min(matrix[1]), np.max(matrix[1])+10050, 50))
        

   
    def import_NPI_DEM(self, path):
        self.NPI_DEM_o = rasterio.open(path)
        self.window = self.NPI_DEM_o.window(np.min(self.x), np.min(self.y), np.max(self.x), np.max(self.y))
        self.dem[1:,1:] = np.flip(self.NPI_DEM_o.read(1, window = self.window), axis = 0) 
        
    def get_vel(self, matrix):
        
        self.vx = np.zeros_like(self.x, dtype='float')
        self.vx[:np.shape(self.x)[0]-int(10000/50),:] = matrix[6]
        self.vx[np.isnan(self.vx)] = 0
        
        self.vy = np.zeros_like(self.x, dtype='float')
        self.vy[:np.shape(self.x)[0]-int(10000/50),:] = matrix[7]
        self.vy[np.isnan(self.vy)] = 0
        
        self.vel = np.zeros_like(self.x, dtype='float')
        self.vel[:np.shape(self.x)[0]-int(10000/50),:] = matrix[8]
        self.vel[np.isnan(self.vel)] = 0
        
    def get_outlines(self, path):
        outline = gpd.read_file(path)
        self.outline = outline.to_crs(self.NPI_DEM_o.crs)
        self.outline_Ko = self.outline.loc[self.outline['RGIId'] == 'RGI60-07.01482', 'geometry']
        self.outline_Kr = self.outline.loc[self.outline['RGIId'] == 'RGI60-07.01464', 'geometry']
        
        ### get mask of Kongsbreen, reproject it and set it to the correct spot in the raster ###
        NPI_mask_Ko, NPI_mask_meta_Ko = rasterio.mask.mask(self.NPI_DEM_o, self.outline_Ko, crop=True)
        NPI_mask_Kr, NPI_mask_meta_Kr = rasterio.mask.mask(self.NPI_DEM_o, self.outline_Kr, crop=True)
        
        NPI_mask_Ko = np.flip(NPI_mask_Ko[0], axis=0)
        NPI_mask_Ko[NPI_mask_Ko<0] = 0
        NPI_mask_Ko[NPI_mask_Ko>0] = 1
        
        NPI_mask_Kr = np.flip(NPI_mask_Kr[0], axis=0)
        NPI_mask_Kr[NPI_mask_Kr<0] = 0
        NPI_mask_Kr[NPI_mask_Kr>0] = 1
        
        #coordinates of upper left corner of cropped (using window) NPI DEM
        x_ul_NPI_DEM = self.window.col_off*50+self.NPI_DEM_o.transform[2]  
        y_ul_NPI_DEM = self.window.row_off*-50+self.NPI_DEM_o.transform[5]
        
        ## amount of pixels that NPI_mask needs to be shifted to be at right spot in the raster
        shift_x_Ko = int((NPI_mask_meta_Ko[2] - x_ul_NPI_DEM)/50)
        shift_y_Ko = int((NPI_mask_meta_Ko[5] - y_ul_NPI_DEM)/50)
        
        shift_x_Kr = int((NPI_mask_meta_Kr[2] - x_ul_NPI_DEM)/50)
        shift_y_Kr = int((NPI_mask_meta_Kr[5] - y_ul_NPI_DEM)/50)
        
        self.mask_Ko = np.zeros_like(self.x, dtype='float')
        self.mask_Ko[(np.shape(self.mask_Ko)[0]-np.shape(NPI_mask_Ko)[0]+shift_y_Ko):(np.shape(self.mask_Ko)[0]+shift_y_Ko), shift_x_Ko:(shift_x_Ko+np.shape(NPI_mask_Ko)[1])]=NPI_mask_Ko

        self.mask_Kr = np.zeros_like(self.x, dtype='float')
        self.mask_Kr[(np.shape(self.mask_Kr)[0]-np.shape(NPI_mask_Kr)[0]+shift_y_Kr):(np.shape(self.mask_Kr)[0]+shift_y_Kr), shift_x_Kr:(shift_x_Kr+np.shape(NPI_mask_Kr)[1])]=NPI_mask_Kr

    def get_dhdt(self, matrix):
        self.dhdt0 = np.zeros_like(self.x, dtype='float')*np.nan
        self.dhdt1 = np.zeros_like(self.x, dtype='float')*np.nan
        self.dhdt0[:np.shape(self.x)[0]-int(10000/50),:] = matrix[3]
        self.dhdt1[:np.shape(self.x)[0]-int(10000/50),:] = matrix[4]
        
        self.mask_Kr = np.ones_like(self.x)
        self.mask_Kr[np.isnan(self.dhdt1)] = 0

        
    def get_mask(self):
        self.retreat_mask = np.zeros_like(self.x, dtype='float')
        self.retreat_mask[np.logical_and(self.dem<100, np.logical_and(self.dhdt0>1, -1*self.mask_Kr+1))] = 1
        self.retreat_mask[self.retreat_mask+self.mask_Kr==2]=0        
        
        self.mask = self.mask_Ko + self.mask_Kr
        self.mask[self.mask>0] = 1
        
        self.dem[self.retreat_mask==1] = 0

    def clean_dhdt(self):
        self.dhdt0[self.mask_Kr==0] = self.dem[self.mask_Kr==0]
        self.dhdt1[self.mask_Kr==0] = self.dem[self.mask_Kr==0]
        
        self.dhdt0[np.isnan(self.dhdt0)] = self.dem[np.isnan(self.dhdt0)]
        self.dhdt1[np.isnan(self.dhdt1)] = self.dem[np.isnan(self.dhdt1)]
        
        self.dhdt = (self.dhdt1 - self.dhdt0)/5
        self.dhdt[np.isnan(self.dhdt)] = 0

        
    def set_dhdt_Ko(self):
        ## fit dhdt for Kongsbreen ##
        poly = np.poly1d(np.polyfit(self.dem[self.mask_Kr == 1], self.dhdt[self.mask_Kr == 1],1))
        dhdt_fit = poly(self.dem)
        dhdt_fit[self.mask_Kr == 1]=0
        dhdt_fit[self.mask_Ko == 0]=0

        ## full dhdt
        self.dhdt = dhdt_fit+self.dhdt
        
    def set_smb(self, path):
        smb_xyz_df = pd.read_excel (path, 0, header=None)
        smb_net_df = pd.read_excel (path, 3, header=None)
        
        smb_x = np.array(smb_xyz_df.loc[:,1])
        smb_y = np.array(smb_xyz_df.loc[:,2])
        smb_z = np.array(smb_xyz_df.loc[:,3])
        
        smb_net_0914 = np.array(smb_net_df.loc[1:,7:12])/100*(10/9)   #convert to m.ice.eq.
        smb_net_0914 = np.nanmean(smb_net_0914, axis=1)
        
        ### interpolate smb with elevation ###
        poly = np.poly1d(np.polyfit(smb_z,smb_net_0914,1))
        self.smb = poly(self.dem)
        self.smb[self.smb<-2] = -2
        self.smb[self.smb>1] = 1
        
    def get_vel_field(self, path):
        ### velocity from Jack ###
        vel_xyz_df = pd.read_excel(path, 1, header=None)
        vel_x = np.array(vel_xyz_df.loc[:,1])
        vel_y = np.array(vel_xyz_df.loc[:,2])
        vel_z = np.array(vel_xyz_df.loc[:,3])
        
        vel_df = pd.read_excel (path, 0, header=None)
        self.vel_field = np.nanmean(np.array(vel_df.loc[3:,range(10,18,2)]), axis=1)
        
    def resample_input(self):
        self.data_res = 50
        self.res = 250
        self.resample = self.data_res/self.res
        
        self.dhdt0 = zoom(self.dhdt0, self.resample)
        self.dhdt1 = zoom(self.dhdt1, self.resample)
        self.dhdt = zoom(self.dhdt, self.resample)
        self.smb = zoom(self.smb, self.resample)
        self.dem = zoom(self.dem, self.resample)
        self.vel = zoom(self.vel, self.resample)
        self.vx = zoom(self.vx, self.resample)
        self.vy = zoom(self.vy, self.resample)
        
        self.mask = np.around(zoom(self.mask, self.resample), 0)
        self.mask_Kr = np.around(zoom(self.mask_Kr, self.resample), 0)
        self.mask_Ko = np.around(zoom(self.mask_Ko, self.resample), 0)
                
        self.x = zoom(self.x, self.resample)
        self.y = zoom(self.y, self.resample) 

    def get_itslive_vel(self, paths):
        itslive_vel_o = rasterio.open(paths[0])
        itslive_vx_o = rasterio.open(paths[1])
        itslive_vy_o = rasterio.open(paths[2])
        
        window_vel = (itslive_vel_o.window(np.min(self.x), np.min(self.y), np.max(self.x), np.max(self.y),1))
        itslive_vel = np.flip(itslive_vel_o.read(1, window = window_vel), axis = 0)
        itslive_vx = np.flip(itslive_vx_o.read(1, window = window_vel), axis = 0)
        itslive_vy = np.flip(itslive_vy_o.read(1, window = window_vel), axis = 0)
        
        self.itslive_vel = np.zeros_like(self.x, dtype='float')
        self.itslive_vel[:,1:] = itslive_vel
        self.itslive_vel[self.itslive_vel<-1e30] = 0
        
        self.itslive_vx = np.zeros_like(self.x, dtype='float')
        self.itslive_vx[:,1:] = itslive_vx
        self.itslive_vx[self.itslive_vx<-1e30] = 0
        
        self.itslive_vy = np.zeros_like(self.x, dtype='float')
        self.itslive_vy[:,1:] = itslive_vy
        self.itslive_vy[self.itslive_vy<-1e30] = 0    
        
    def filter_dhdt(self):
        ### filter outliers in dhdt ###
        dhdt_full_new = np.zeros_like(self.dhdt)
        for i in range(np.shape(self.dhdt)[0]):
            for j in range(np.shape(self.dhdt)[1]):
                if self.mask[i, j]==0:
                        dhdt_full_new[i, j] = 0
                else:
                    y_f, x_f = np.ogrid[:np.shape(self.dhdt)[0], :np.shape(self.dhdt)[1]]
                    circle = (y_f - i) ** 2 + (x_f - j) ** 2 <= 20
                    
                    local_med = np.median(self.dhdt[np.logical_and(circle, self.mask==1)])
                    local_std = np.std(self.dhdt[np.logical_and(circle, self.mask==1)])
                    
                    if self.dhdt[i,j] >  local_med + 1* local_std:
                        dhdt_full_new[i, j]= local_med + 1* local_std
                    elif self.dhdt[i,j] < local_med - 1* local_std:
                        dhdt_full_new[i, j]= local_med - 1* local_std              
                    else:
                        dhdt_full_new[i,j] = self.dhdt[i,j]
        self.dhdt = dhdt_full_new
        
    def set_parameters(self, ice_temp=273, ice_density = 900., secpera = 31556926., g = 9.81):
        self.A = 1.733e3*np.exp(-13.9e4/(8.3*ice_temp))
        self.ice_density = ice_density
        self.secpera = secpera
        self.g = g
    
    def build_input(self):

        DEM_VEL_mat = scipy.io.loadmat('./kronebreen/HDF_2021_10_06.mat')['HDF'][0,0]
        self.set_xy(DEM_VEL_mat)

        self.reset_shape()

        self.import_NPI_DEM("./kronebreen/S0_DTM50.tif")

        self.get_vel(DEM_VEL_mat)

        self.get_outlines('./kronebreen/rgi_outlines/07_rgi60_Svalbard.shp')

        self.get_dhdt(DEM_VEL_mat)

        self.get_mask()

        self.clean_dhdt()

        self.set_dhdt_Ko()

        self.set_smb('./kronebreen/HDF_mass_balance.xlsx')

        self.get_vel_field('./kronebreen/HDF_stake_velocities.xlsx')

        self.resample_input()

        self.get_itslive_vel(['./kronebreen/vel_ITSLIVE_resample.tif', "./kronebreen/vx_ITSLIVE_proj.tif", "./kronebreen/vy_ITSLIVE_proj.tif"])

        self.filter_dhdt()

        self.set_parameters()

        self.check_shape()
        
        
class radar_data():
    
    def __init__(self):
        self.x = []
        self.y = []
        self.rad = []
        self.x_arr = []
        self.mask_arr = []
        self.rad_arr = []
        
    def rad_as_array(input):
        #find the index that is associated with the radar x-y coordiantes
        gr = np.meshgrid(range(np.shape(input.dem)[1]), range(np.shape(input.dem)[0]))
        index_y = get_nearest(input.x,input.y,gr[1], self.x, self.y) 
        index_x = get_nearest(input.x,input.y,gr[0],self.x, self.y)
        grid_points = list(zip(index_y, index_x))
        
        # take the mean of all measured thicknesses that fall on one grid point
        df_place = pd.DataFrame({"inds":list(zip(index_y, index_x)),"rad":self.rad[:,0], "x_rad":self.x[:,0], "y_rad": self.y[:,0]})
        df_place = df_place.groupby('inds').mean()
        inds = np.array(df_place.index)
        bed_loc = df_place['rad']
        x_rad_sample = df_place['x_rad']
        self.mask_arr = np.zeros_like(input.x, dtype='float')
        self.bed_arr = np.zeros_like(input.x, dtype='float')
        self.x_arr = np.zeros_like(input.x, dtype='float')
        
        n = int(len(inds)/2)
        random_index = np.random.choice(len(inds), n, replace=False)  
        
        # modify for-statement to sample measured bed elevations according to study question
        for i in inds:
            self.mask_arr[i]=1
            self.bed_arr[i] = bed_loc[i]
            self.x_arr[i] = x_rad_sample[i]
        
        
    def check_shape(self):
        check_shape(self.x, self.y)
        check_shape(self.x, self.rad)
    
    
def check_shape(x, data, flag = 0):
    if np.shape(x) == np.shape(data):
        print('all good')
    else:
        for d in data:
            if len(d)>0 and np.shape(d) != np.shape(x):
                    raise ValueError('not correct shape of input data') 

def create_init(input):
    WRIT_FILE = 'kronebreen_kongsbreen_initialSetup.nc'

    ### CONSTANTS ###
    
    ny, nx = np.shape(input.x)
    Lx = nx * input.res  # in m
    Ly =  ny * input.res # in m
    
    x_in = np.linspace(-Lx/2, Lx/2, nx)
    y_in = np.linspace(-Ly/2, Ly/2, ny)
    
    B_rec = input.dem
    B_init = deepcopy(B_rec)
    ice_surface_temp = np.ones((ny, nx))*input.ice_temp
    #M_refs = np.nan_to_num(M_refs)
    #M_refs *= mask
    
    h_rec = input.dem - B_rec
    
    ##### define variables, set attributes, write data #####
    # format: ['units', 'long_name', 'standard_name', '_FillValue', array]
    
    vars = {'y':    ['m',
                     'y-coordinate in Cartesian system',
                     'projection_y_coordinate',
                     None,
                     y_in],
            'x':    ['m',
                     'x-coordinate in Cartesian system',
                     'projection_x_coordinate',
                     None,
                     x_in],
            'thk':  ['m',
                     'ice thickness',
                     'land_ice_thickness',
                     1.0,
                     h_rec],
            'topg': ['m',
                     'bedrock surface elevation',
                     'bedrock_altitude',
                     0.0,
                     B_rec],
            'ice_surface_temp': ['K',
                                 'annual mean air temperature at ice surface',
                                 'surface_temperature',
                                 273,
                                 ice_surface_temp],
            'climatic_mass_balance': ['kg m-2 year-1',
                                      'mean annual net ice equivalent accumulation rate',
                                      'land_ice_surface_specific_mass_balance_flux',
                                      0,
                                      input.smb * input.ice_density * input.mask]
            }
    
    ##### define dimensions in NetCDF file #####
    def create_nc_input(vars, WRIT_FILE):
        ncfile = NC(WRIT_FILE, 'w', format='NETCDF3_CLASSIC')
        xdim = ncfile.createDimension('x', nx)
        ydim = ncfile.createDimension('y', ny)
        
        for name in list(vars.keys()):
            [_, _, _, fill_value, data] = vars[name]
            if name in ['x', 'y']:
                var = ncfile.createVariable(name, 'f4', (name,))
            else:
                var = ncfile.createVariable(name, 'f4', ('y', 'x'), fill_value=fill_value)
            for each in zip(['units', 'long_name', 'standard_name'], vars[name]):
                if each[1]:
                    setattr(var, each[0], each[1])
            var[:] = data
        
        # finish up
        ncfile.close()
        print("NetCDF file ", WRIT_FILE, " created")
        print('')
        
    create_nc_input(vars, WRIT_FILE)
    
    
def scale(x):
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

def gauss_filter(U, sigma, truncate):

    V=U.copy()
    V[np.isnan(U)]=0
    VV=ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)

    Z=VV/WW
    return Z

def get_nearest(x_ref,y_ref,reference, x_dat, y_dat):
    grid_temp = []
    for i in range(len(x_dat)):
        abslat = np.abs(x_ref-x_dat[i])
        abslon= np.abs(y_ref-y_dat[i])

        c = np.maximum(abslon,abslat)
        latlon_idx = np.argmin(c)
        grid_temp.append(reference.flat[latlon_idx])
    return grid_temp

#def neighbors(a, radius, rowNumber, columnNumber):
#     return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
#                for j in range(columnNumber-1-radius, columnNumber+radius)]
#                    for i in range(rowNumber-1-radius, rowNumber+radius)]
 
def create_script(forward_or_iteration, nx, ny):
    print("""#!/bin/bash
    ###### run script for experiment synthetic1 ######""")
    print('# build the PISM command')
    print('set -e #exit on error')
    print('')
    print('NN="$1"')
    print('CLIMATEFILE="$2"')
    print('DURATION=$3')
    print('OUTNAME=$4')
    print('OPT5=$5')
    print('RUNTIME="-ys 0 -ye $DURATION"')
    
    print('')
    print('CLIMATE="-surface given -surface_given_file $CLIMATEFILE"')
    print('grid="-Mx {} -My {} -Mz 50 -Mbz 1 -Lz 1500 -Lbz 1"'.format(nx, ny))
    print('PHYS="-stress_balance ssa+sia -sia_flow_law isothermal_glen -ssa_flow_law isothermal_glen"')# -basal_resistance.beta_lateral_margin 0 -stress_balance.ssa.fd.lateral_drag.viscosity 0"')
    print('THERMAL="-energy none"')
    #print('OCEAN="-dry"')
    print('CONF="-config_override kronebreen_kongsbreen_conf.nc"')

    # power law sliding relation t_b=-C*|u|^(m-1)*u --> doesn't require thermal model
    print('SLIDING="-pseudo_plastic -pseudo_plastic_q 0.2 -pseudo_plastic_uthreshold 3.1556926e7 -yield_stress constant -tauc 1e5"')
    
    print('echo')
    print('echo "# ======================================================================="')
    print('echo "# initialize Kronebreen"')
    print('echo "#  $NN processors, $DURATION a run, 50 km grid, $CLIMATEFILE, $4"')
    print('echo "# ======================================================================="')
    
    print('')
    print('PISM_MPIDO="mpiexec -n "')
    
    print('')
    print('PISM_BIN=/home/thomas/pism/bin')
    print('PISM_EXEC="pismr"')
    print('EXVARS="temppabase,tempicethk_basal,velsurf_mag,mask,thk,usurf,velbase_mag,enthalpybase,bwat,strain_rates"')
    
    print('')
    print('PISM="${PISM_BIN}/${PISM_EXEC}"')
    
    print('')
    print('EXSTEP=100')
    print('TSNAME=ts_$OUTNAME')
    print('TSTIMES=0:yearly:$DURATION')
    print('EXNAME=ex_$OUTNAME')
    print('EXTIMES=0:$EXSTEP:$DURATION')
    print('DIAGNOSTICS="-ts_file $TSNAME -ts_times $TSTIMES -extra_file $EXNAME -extra_times $EXTIMES -extra_vars $EXVARS -o_size big"')
    
    print('DIAGNOSTICS_ITER="-save_file s_$OUTNAME -save_force_output_times -o_size big -extra_vars $EXVARS"')
    
    print('')
    if forward_or_iteration == 'forward':
        print('cmd="$PISM_MPIDO $NN $PISM -i $CLIMATEFILE -bootstrap ${grid} $CONF $SLIDING $THERMAL $OCEAN $RUNTIME $CLIMATE $PHYS $DIAGNOSTICS -o $OUTNAME"')
    elif forward_or_iteration == 'iteration':
        print('cmd="$PISM_MPIDO $NN $PISM -i $CLIMATEFILE $CONF $ENHANCE $SLIDING $THERMAL $OCEAN $RUNTIME $CLIMATE $PHYS -o $OUTNAME"')
    
    print('')
    print('echo')
    print('$cmd')
    
def launch_init(nx, ny):
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open('kronebreen_kongsbreen_initialize.sh', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        create_script('forward', nx, ny)
        sys.stdout = original_stdout # Reset the standard output to its original value
        f.close()
        
    cmd = ['chmod', '+x', 'kronebreen_kongsbreen_initialize.sh']
    subprocess.call(cmd)
    cmd = ['./kronebreen_kongsbreen_initialize.sh', '4', 'kronebreen_kongsbreen_initialSetup.nc', '1', 'kronebreen_kongsbreen_output.nc > kronebreen_kongsbreen_output_log.txt']
    subprocess.call(cmd)
    
import richdem as rd
import math

def stagger(x):
    x_stag = np.zeros_like(x)
    x_stag[1:-1,1:-1] = 0.25*(x[1:-1,1:-1]+x[1:-1,0:-2]+x[0:-2,0:-2]+x[0:-2,1:-1])
    return x_stag

def x_deriv(x, res):
    dxdx = np.zeros_like(x, dtype='float')
    dxdx[0:-2,0:-2] = 0.5 * (x[1:-1,0:-2]-x[1:-1,1:-1] + x[0:-2,0:-2] - x[0:-2,1:-1])/res
    return dxdx
def y_deriv(y, res):
    dydy = np.zeros_like(y, dtype='float')
    dydy[0:-2,0:-2] = 0.5*(y[0:-2,1:-1]-y[1:-1,1:-1] + y[0:-2,0:-2] - y[1:-1,0:-2])/res
    return dydy
    
def initial_bed_guess(input, n = 20):
    dH = (np.nanmax(input.dem[input.mask==1])-np.nanmin(input.dem[input.mask==1]))/1000 #in km
    
    tau = 0.005+1.598*dH-0.435*dH**2  #Haeberli and Hoelzle
    
    S_stag = stagger(input.dem)
    dhdx = x_deriv(S_stag, input.res)
    dhdy = y_deriv(S_stag, input.res)
    input.dem_slope = np.sqrt(dhdx**2+dhdy**2)
    
    input.dem_slope[input.dem_slope<0.015] = 0.015
    input.dem_sin_slope = np.sin(np.arctan(input.dem_slope))
    H = ((tau*.5)*1e5)/(input.dem_sin_slope*input.g*input.ice_density)
    H_smooth = deepcopy(H)
    sin_slope_smooth = deepcopy(input.dem_sin_slope)
    for i in range(n):
        sin_slope_smooth = smooth_stress_coupling(H_smooth, input.dem_sin_slope, input.mask, 4, 8,2, input.res)
        H_smooth =(tau*.5)*1e5/(sin_slope_smooth*input.g*input.ice_density)
    
    input.initial_thickness = H_smooth
    
    return input

    
def smooth_stress_coupling(H, field, mask, scl, max_scl, min_scl, res):
    field_new = np.zeros_like(field)
    for i in range(np.shape(field)[1]):
        for j in range(np.shape(field)[0]):
            if mask[j,i]==0:
                field_new[j,i] = field[j,i]
            else:
                x, y = np.ogrid[:np.shape(H)[0], :np.shape(H)[1]]
                coupling_length = scl*(H[j,i]/res)**2
                circle = (y - i) ** 2 + (x - j) ** 2 <= (min(max(coupling_length, min_scl),max_scl)/2)**2
                field_new[j,i]= np.mean(field[np.logical_and(circle, mask==1)])
    return field_new
         
def cal_diffusivity(thk, A,g,rho, alpha, mask):
    T=(2*A*(g*rho)**3)/5
    D=mask*T*thk**5*(alpha)**2
    return D, T
    
               
class model():
    def __init__(self, input):
        self.it_fields = self.it_fields_class(input)
        self.it_parameters = self.it_parameters_class(input)
        self.it_products = self.it_products_class()
        self.series = self.series_class()
        self.file_locations = self.file_locations_class()
        self.warnings = []
        
    class it_fields_class:
        def __init__(self, input):
            self.S_ref = input.dem
            self.S_rec = copy(self.S_ref)                       # no smoothing applied in standard initialization
            self.initial_thickness = initial_bed_guess(input, 20).initial_thickness
            self.B_rec = self.S_rec - self.initial_thickness           # no smoothing applied in standard initialization
            self.tauc_rec = (500+self.B_rec)*1e3
            self.dh_ref = input.dhdt
            self.mask = input.mask
            self.vel_mes = input.itslive_vel
            self.smb = input.smb
            self.B_init = copy(self.B_rec)
            self.S_init = copy(self.S_rec)
          
    class series_class:           
        def __init__(self):
            self.B_rec_all = []
            self.dh_all = []
            self.misfit_all = []
            self.B_misfit_vs_iter=[]
            self.dh_misfit_vs_iter=[]
            self.S_rec_all = []
            self.vel_all = []
            self.tauc_recs = []
       
    class it_parameters_class:      
        def __init__(self, input):
            self.pmax = 2999
            self.dt = 0.1
            self.beta = 0.5
            self.shift = 0.3
            self.delta_surf = 0.025
            self.p_friction = 10000
            self.diff_lim = 1e-1
            self.n_cores = 7
            self.A = 3.9565534675428266e-24
            self.g = 9.81
            self.ice_density = 900
            self.smooth_surf_in = (1,3)
            self.smooth_B_in = (.6, .3)
            self.res = copy(input.res)
            self.max_time = 10
            self.tauc_scale = 1
            
    class it_products_class:     
        def __init__(self):
            self.vel_mod = []
            self.u_mod = []
            self.v_mod = []
            self.misfit = []
            self.h_rec = []
            self.dh_rec = []
            self.slope_iter = []
            self.D_iter = []
            self.vel_mismatch = []
            self.h_old = []    
            self.H_rec = []
            self.start_time = time.time()
        
    class file_locations_class():     
        def __init__(self):
            self.it_out = 'kronebreen_kongsbreen_iteration_out.nc'
            self.it_in = 'kronebreen_kongsbreen_iteration_in.nc'
            self.it_log = 'kronebreen_kongsbreen_iteration_log.txt'
            self.it_script = 'kronebreen_kongsbreen_iteration_script.sh' 
            self.init_output = 'kronebreen_kongsbreen_output.nc'
            
    def smooth_SandB(self, input):
        self.it_fields.S_rec[self.it_fields.mask==0] = np.nan
        self.it_fields.S_rec = gauss_filter(self.it_fields.S_rec, self.it_parameters.smooth_surf_in[0], self.it_parameters.smooth_surf_in[1])
        self.it_fields.S_rec[self.it_fields.mask==0] = self.it_fields.S_ref[self.it_fields.mask==0]
        
        D = cal_diffusivity(self.it_fields.initial_thickness, self.it_parameters.A, self.it_parameters.g, self.it_parameters.ice_density, input.dem_slope, self.it_fields.mask)
        self.it_fields.initial_thickness[D[0]>self.it_parameters.diff_lim] = (self.it_parameters.diff_lim/(D[1]*(input.dem_slope[D[0]>self.it_parameters.diff_lim])**2))**(1/5)
        
        self.it_fields.B_rec[self.it_fields.mask==0] = np.nan
        self.it_fields.B_rec = gauss_filter(self.it_fields.B_rec, self.it_parameters.smooth_B_in[0], self.it_parameters.smooth_B_in[1])
        self.it_fields.B_rec[np.logical_or(self.it_fields.B_rec > self.it_fields.S_rec, self.it_fields.mask==0)] = self.it_fields.S_rec[np.logical_or(self.it_fields.B_rec > self.it_fields.S_rec, self.it_fields.mask==0)]
        
        
    def calc_slope(self):
        S_rec_stag = stagger(self.it_fields.S_rec)
        dhdx = x_deriv(S_rec_stag, self.it_parameters.res)
        dhdy = y_deriv(S_rec_stag, self.it_parameters.res)
        self.it_products.slope_iter = np.sqrt(dhdx**2 + dhdy**2)
        
    def calc_misfit(self):
        self.it_products.misfit = shift(self.it_products.dh_rec - self.it_fields.dh_ref, self.it_products.u_mod, self.it_products.v_mod, self.it_parameters.shift) * self.it_fields.mask
        
    def calc_dh_rec(self):
        self.it_products.dh_rec = (self.it_products.h_rec - self.it_products.h_old)/self.it_parameters.dt
        
    def get_vels_mod(self):
        self.it_products.u_mod = get_nc_data(self.file_locations.it_out, 'uvelsurf', 0)
        self.it_products.v_mod = get_nc_data(self.file_locations.it_out, 'vvelsurf', 0)
        self.it_products.vel_mod = get_nc_data(self.file_locations.it_out, 'velsurf_mag', 0)
        
    def update_bed(self):
        self.it_fields.B_rec -= self.it_parameters.beta * self.it_products.misfit
        
    def update_surface(self):
        self.it_fields.S_rec[np.logical_and(self.it_fields.mask == 1, self.it_products.h_rec>20)] += self.it_parameters.delta_surf * self.it_products.misfit[np.logical_and(self.it_fields.mask == 1, self.it_products.h_rec>20)]
                                                                                                                    
    def correct_for_diffusivity(self):
        self.it_products.H_rec = self.it_fields.S_rec - self.it_fields.B_rec
        self.it_products.D_iter = cal_diffusivity(self.it_products.H_rec, self.it_parameters.A, self.it_parameters.g, self.it_parameters.ice_density, self.it_products.slope_iter, self.it_fields.mask)
        self.it_products.H_rec[self.it_products.D_iter[0]>self.it_parameters.diff_lim] = (1e-1/(self.it_products.D_iter[1]*(self.it_products.slope_iter[self.it_products.D_iter[0]>self.it_parameters.diff_lim])**2))**(1/5)
        self.it_fields.B_rec = self.it_fields.S_rec - self.it_products.H_rec
        
    def mask_fields(self):
        self.it_fields.B_rec[self.it_fields.B_rec>self.it_fields.S_rec] = self.it_fields.S_rec[self.it_fields.B_rec>self.it_fields.S_rec]     
        self.it_fields.B_rec[self.it_fields.B_rec>self.it_fields.S_ref] = self.it_fields.S_rec[self.it_fields.B_rec>self.it_fields.S_ref]
        self.it_fields.B_rec[self.it_fields.mask==0] = self.it_fields.S_ref[self.it_fields.mask==0]
        self.it_fields.S_rec[self.it_fields.mask==0] = self.it_fields.S_ref[self.it_fields.mask==0]
        
    def append_series(self):
        self.series.B_rec_all.append(self.it_fields.B_rec)
        self.series.dh_all.append(self.it_products.dh_rec)
        self.series.misfit_all.append(self.it_products.misfit)
        self.series.B_misfit_vs_iter.append(np.mean(abs(self.it_fields.B_rec - self.it_fields.B_init)))
        self.series.dh_misfit_vs_iter.append(np.mean(abs(self.it_products.dh_rec[self.it_fields.mask==1] - self.it_fields.dh_ref[self.it_fields.mask==1])))
        self.series.S_rec_all.append(self.it_fields.S_rec)
        self.series.vel_all.append(self.it_products.vel_mod)
        self.series.tauc_recs.append(self.it_fields.tauc_rec)
        
    def update_tauc(self):
        self.it_products.vel_mismatch = np.maximum(np.minimum((self.it_products.vel_mod - self.it_fields.vel_mes)/self.it_fields.vel_mes, 1), -1)
        self.it_fields.tauc_rec += self.it_products.vel_mismatch * self.it_fields.tauc_rec * self.it_parameters.tauc_scale
                
    def update_nc(self):
        nc_updated = NC(self.file_locations.it_in, 'r+')
        nc_updated['topg'][0,:,:] = self.it_fields.B_rec
        nc_updated['thk'][0,:,:]=self.it_products.h_old
        nc_updated['tauc'][0,:,:]= self.it_fields.tauc_rec
        nc_updated['climatic_mass_balance'][0,:,:] = self.it_fields.smb
        nc_updated.close()

    def short_forward(self):
        cmd = ['./{}'.format(self.file_locations.it_script), str(self.it_parameters.n_cores), self.file_locations.it_in, str(self.it_parameters.dt), self.file_locations.it_out]
        subprocess.call(cmd, stdout = open(self.file_locations.it_log, 'a'))

    def create_it_script(self):
        original_stdout = sys.stdout # Save a reference to the original standard output
        with open(self.file_locations.it_script, 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            create_script('iteration', np.shape(self.it_fields.S_ref)[0], np.shape(self.it_fields.S_ref)[1])
            sys.stdout = original_stdout # Reset the standard output to its original value
            f.close()
        
    def iterate(self, input):
        self.smooth_SandB(input)
        self.create_it_script()
        subprocess.call(['cp', self.file_locations.init_output, self.file_locations.it_out])
        
        self.it_products.start_time = time.time()
        for p in range(self.it_parameters.pmax):
            self.it_products.h_old = self.it_fields.S_rec - self.it_fields.B_rec     
            subprocess.call(['cp', self.file_locations.it_out, self.file_locations.it_in])
            self.update_nc()
            self.short_forward()
            self.it_products.h_rec = get_nc_data(self.file_locations.it_out, 'thk', 0)
            self.calc_dh_rec()
            self.get_vels_mod()
            self.calc_misfit()
            self.update_bed()
            self.update_surface()
            self.calc_slope()
            self.correct_for_diffusivity()
            self.mask_fields()
            self.append_series()
            if p>0 and p%self.it_parameters.p_friction == 0:
                self.update_tauc()
            if time.time() > self.it_products.start_time + self.it_parameters.max_time * 60 * 60:
                self.warnings.append('run did not finish in designated max time')
                break


################## AUTO SCRIPT ####################


print('loading input data')
data = input_data()
data.build_input()

print('populating model')
md = model(data)

runs = [
        {'it_parameters.dt': 0.2, 'it_parameters.max_time': 30}, #bit longer dt
        {'it_parameters.dt': 0.5, 'it_parameters.max_time': 30},   #somewhat longer dt
        {'it_parameters.dt': 1, 'it_parameters.max_time': 50}, #even longer dt
        {'it_parameters.dt': 5, 'it_parameters.max_time': 80}, #very long dt
        ]

        


for element in runs:
    if isinstance(element, dict) == False:
        raise ValueError('{} not in correct format'.format(element))
        
save_strings = []
for run in runs:
    save_string = ''
    for string in run:
        if isinstance(run[string], int):
            save_string += string + '_' + '{}'.format(run[string]) + '_'
        elif isinstance(run[string], np.ndarray):
            save_string += string + '_'
        elif isinstance(run[string], float):
            save_string += string + '_' + '{}'.format(round(run[string], 2)) + '_'
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
        with open('KK_auto_{}_{}.pkl'.format(save_strings[i], i), 'wb') as outp:
            pickle.dump(md, outp, pickle.HIGHEST_PROTOCOL)
        subprocess.call(['cp', md.file_locations.it_out, './auto_ncs/KK_auto_{}_{}.nc'.format(save_strings[i], i)])
        print('saving done, this was run {} out of {} runs'.format(i+1, len(runs)))
        md.__init__(data)
    except:
        print('something went wrong in iteration or during saving')
        md.warnings.append('something went wrong in iteration or during saving')
        md.__init__(data)
        continue
