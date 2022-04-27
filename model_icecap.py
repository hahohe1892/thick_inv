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

class input_data():
    
    def __init__(self):
            self.x, self.y = np.meshgrid(1,1)
            self.smb = np.zeros_like(self.x, dtype='float')
            self.dem = np.zeros_like(self.x, dtype='float')
            self.dhdt = np.zeros_like(self.x, dtype='float')
            self.mask = np.zeros_like(self.x, dtype='float')
    
    def check_shape(self):
        check_shape(self.x, [self.dem, self.smb, self.dhdt, self.mask])

    def set_xy(self):
        self.Lx = 2 * 25e3  # in km
        self.Ly = 2 * 25e3  # in km

        self.dx, self.dy = 1e3,1e3
        
        self.ny = int(np.floor(self.Lx / self.dx) + 1)  # make it an odd number
        self.nx = int(np.floor(self.Ly / self.dy) + 1)  # make it an odd number
        
        self.x = np.linspace(-self.Lx, self.Lx, self.nx)
        self.y = np.linspace(-self.Ly, self.Ly, self.ny)
   
    def reset_shape(self):
        self.dem = np.zeros_like(self.x, dtype='float')
        self.smb = np.zeros_like(self.x, dtype='float')
        self.dhdt = np.zeros_like(self.x, dtype='float')
        self.mask = np.zeros_like(self.x, dtype='float')
        
    def get_setup(self):
        self.dem = np.ones((self.ny, self.nx))
        self.topg = np.zeros((self.ny, self.nx))
        self.tauc = np.zeros((self.ny, self.nx))
        self.smb = np.zeros((self.ny, self.nx))
        self.thk = np.ones((self.ny,self.nx))
        
        self.base = np.ones((self.ny, self.nx)) * (self.x/100+500)
        
        for i in range(0, self.nx):
            for j in range(0, self.ny):
                dist = ((self.x[i])**2+(self.y[j])**2)**0.5
                dist2a=((self.x[i]-self.Lx/5)**2+(self.y[j]-self.Ly/5)**2)**0.5
                dist2b=((self.x[i]+self.Lx/5)**2+(self.y[j]+self.Ly/5)**2)**0.5
                dist2c = ((2*(self.x[i])+self.Lx/5)**2+(2*(self.y[j])-self.Ly/5)**2)**0.5
                dist2d = ((2*(self.x[i])-self.Lx/5)**2+(2*(self.y[j])+self.Ly/5)**2)**0.5
                self.topg[i, j] = (np.maximum(500*(1-dist2a*5/self.Lx),0)+np.maximum(500*(1-dist2b*5/self.Lx),0)+np.maximum(500*(1-dist2c*5/self.Lx),0)+np.maximum(500*(1-dist2d*5/self.Lx),0))
                self.smb[i, j] = 5 *(1-(dist*2/self.Lx))
                self.dem[i,j] = self.topg[i,j]+1
                
        self.tauc = np.ones((self.ny,self.nx))*5e7
        for i in range(0,self.nx):
            for j in range(0, self.ny):
                if j<=24:
                    self.tauc[i,j] -= 3e7*np.exp((-(self.x[i])**2)/(2*7000**2))
        self.dem = self.topg+self.thk
        
                
    def set_parameters(self, ice_temp=268, ice_density = 900., secpera = 31556926., g = 9.81):
        self.ice_temp = ice_temp
        self.A = 1.733e3*np.exp(-13.9e4/(8.3*ice_temp))
        self.ice_density = ice_density
        self.secpera = secpera
        self.g = g
    
    def build_input(self):

        self.set_xy()
        
        self.get_setup()

        self.set_parameters()

        self.smb *= self.ice_density

    
def check_shape(x, data, flag = 0):
    if np.shape(x) == np.shape(data):
        print('all good')
    else:
        for d in data:
            if len(d)>0 and np.shape(d) != np.shape(x):
                    raise ValueError('not correct shape of input_data data') 

##### define dimensions in NetCDF file #####
def create_nc_input_data(vars, WRIT_FILE, nx, ny):
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
                        
def create_init(input_data):
    WRIT_FILE = 'icecap_initialSetup.nc'
    ##### define variables, set attributes, write data #####
    # format: ['units', 'long_name', 'standard_name', '_FillValue', array]
    
    vars = {'y':    ['m',
                     'y-coordinate in Cartesian system',
                     'projection_y_coordinate',
                     None,
                     input_data.y],
            'x':    ['m',
                     'x-coordinate in Cartesian system',
                     'projection_x_coordinate',
                     None,
                     input_data.x],
            'thk':  ['m',
                     'floating ice shelf thickness',
                     'land_ice_thickness',
                     None,
                     input_data.thk],
            'topg': ['m',
                     'bedrock surface elevation',
                     'bedrock_altitude',
                     None,
                     input_data.topg],
            'climatic_mass_balance': ['kg m-2 year-1',
                                      'mean annual net ice equivalent accumulation rate',
                                      'land_ice_surface_specific_mass_balance_flux',
                                      None,
                                      input_data.smb],
            'ice_surface_temp': ['K',
                                 'annual mean air temperature at ice surface',
                                 'surface_temperature',
                                 273,
                                 input_data.ice_temp],
            'tauc': ['Pa',
                     'yield stress for basal till (plastic or pseudo-plastic model)',
                     'yield stress',
                     None,
                     tauc],
            }
    
    create_nc_input_data(vars, WRIT_FILE, input_data.nx, input_data.ny)

    
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
 
def create_script(forward_or_iteration, nx, ny):
    print("""#!/bin/bash
    ###### run script for experiment icecap ######""")
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
    print('grid="-Mx 51 -My 51 -Mz 30 -Mbz 1 -Lz 8000 -Lbz 1 -grid.recompute_longitude_and_latitude false"')
    print('PHYS="-stress_balance ssa+sia -ssa_flow_law isothermal_glen"')
    #print('PHYS="-stress_balance blatter"')
    print('THERMAL="-energy none"')
    print('CONF="-config_override icecap_conf.nc"')

    
    # power law sliding relation t_b=-C*|u|^(m-1)*u --> doesn't require thermal model
    print('SLIDING="-pseudo_plastic -pseudo_plastic_q 0.33333 -pseudo_plastic_uthreshold 3.1556926e7 -yield_stress constant"')
    
    print('echo')
    print('echo "# ======================================================================="')
    print('echo "# create icecap"')
    print('echo "#  $NN processors, $DURATION a run, 50 km grid, $CLIMATEFILE, $4"')
    print('echo "# ======================================================================="')
    
    print('')
    print('PISM_MPIDO="mpiexec -n "')
    
    print('')
    print('PISM_BIN=/home/thomas/pism/bin')
    print('PISM_EXEC="pismr"')
    print('EXVARS="temppabase,tempicethk_basal,velsurf_mag,mask,thk,usurf,velbase_mag, uvel, vvel"')
    
    print('')
    print('PISM="${PISM_BIN}/${PISM_EXEC}"')
    
    print('')
    print('EXSTEP=100')
    print('TSNAME=ts_$OUTNAME')
    print('TSTIMES=0:yearly:$DURATION')
    print('EXNAME=ex_$OUTNAME')
    print('EXTIMES=0:$EXSTEP:$DURATION')
    print('DIAGNOSTICS="-ts_file $TSNAME -ts_times $TSTIMES -extra_file $EXNAME -extra_times $EXTIMES -extra_vars $EXVARS"')
    
    print('DIAGNOSTICS_ITER="-save_file s_$OUTNAME -save_times $OPT5 -save_force_output_times"')

    
    print('')
    if forward_or_iteration == 'forward':
        print('cmd="$PISM_MPIDO $NN $PISM -i $CLIMATEFILE -bootstrap ${grid} $SLIDING $THERMAL $CONF $RUNTIME $CLIMATE $PHYS $DIAGNOSTICS -o $OUTNAME"')
    elif forward_or_iteration == 'iteration':
        print('cmd="$PISM_MPIDO $NN $PISM -i $CLIMATEFILE $CONF $DIF $ENHANCE $SLIDING $THERMAL $OCEAN $RUNTIME $CLIMATE $PHYS -o $OUTNAME"')
    
    print('')
    print('echo')
    print('$cmd')
    
def launch_init(nx, ny):
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open('icecap_build_script.sh', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        create_script('forward')
        sys.stdout = original_stdout # Reset the standard output to its original value
        f.close()
        
def build_icecap():  
    cmd = ['chmod', '+x', 'icecap_build_script.sh']
    subprocess.call(cmd)
    cmd = ['./icecap_build_script.sh', '4', 'icecap_initialSetup.nc', '10000', 'icecap_output.nc > icecap_output_log.txt']
    subprocess.call(cmd)
    
def retrieve_output():
    S_ref = get_nc_data('icecap_output.nc', 'usurf', 0)
    h_ref = get_nc_data('icecap_output.nc', 'thk', 0)
    vel_ref = get_nc_data('icecap_output.nc', 'velsurf_mag', 0).data
    mask = get_nc_data('icecap_output.nc', 'mask', 0)/2
    
    return S_ref, h_ref, vel_ref, mask

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
    
               
class model():
    def __init__(self, input_data):
        self.it_fields = self.it_fields_class(input_data)
        self.it_parameters = self.it_parameters_class(input_data)
        self.it_products = self.it_products_class()
        self.series = self.series_class()
        self.file_locations = self.file_locations_class()
        self.warnings = []
        
    class it_fields_class:
        def __init__(self, input_data):
            self.S_ref = retrieve_output()[0]
            self.S_rec = copy(self.S_ref)
            self.B_rec = np.zeros_like(self.S_ref)      
            self.tauc = copy(input_data.tauc)
            self.tauc_rec = np.ones_like(self.S_ref)*5e7
            self.dh_ref = np.zeros_like(self.S_ref)
            self.mask = retrieve_output()[-1]
            self.vel_mes = retrieve_output()[2]
            self.smb = input_data.smb
            self.B_init = copy(self.B_rec)
            self.S_init = copy(self.S_rec)
            self.tauc_init = copy(self.tauc_rec)
            self.h_ref = retrieve_output()[1]
            self.B_ref = input_data.topg
            self.x, self.y = input_data.x, input_data.y
          
    class series_class:           
        def __init__(self):
            self.B_rec_all = []
            self.dh_all = []
            self.S_rec_all = []
            self.vel_all = []
            self.tauc_rec_all = []
            self.B_misfit_vs_iter=[]
            self.dh_misfit_vs_iter=[]
            self.vel_misfit_vs_iter = []
            self.tauc_misfit_vs_iter=[]

       
    class it_parameters_class:      
        def __init__(self, input_data):
            self.pmax = 7000
            self.dt = 0.1
            self.beta = 1
            self.shift = 0.3
            self.delta_surf = 0.02
            self.p_friction = 1000
            self.bw = 3
            self.n_cores = 4
            self.A = input_data.A
            self.g = input_data.g
            self.ice_density = input_data.ice_density
            self.ice_temp = input_data.ice_temp
            self.tauc_scale = 1
            self.max_time = 5  #hours
            
    class it_products_class:     
        def __init__(self):
            self.vel_mod = []
            self.u_mod = []
            self.v_mod = []
            self.misfit = []
            self.h_rec = []
            self.dh_rec = []
            self.vel_mismatch = []
            self.h_old = []    
            self.H_rec = []
            self.mask_iter = []
            self.start_time = time.time()
        
    class file_locations_class():     
        def __init__(self):
            self.it_out = 'icecap_iteration_out.nc'
            self.it_in = 'icecap_iteration_in.nc'
            self.it_log = 'icecap_iteration_log.txt'
            self.it_script = 'icecap_iteration_script.sh' 
            self.boot_in = 'icecap_bootstrap_in.nc'
            self.boot_out = 'icecap_bootstrap_out.nc'
            self.boot_script = 'icecap_bootstrap.sh'

    def copy_initial_state(self): #reinitialize all fields that depend on other fields, because if other fields are changed, these should be changed as well
        self.it_fields.S_rec = copy(self.it_fields.S_ref)
        self.it_fields.B_init = copy(self.it_fields.B_rec)
        self.it_fields.tauc_init = copy(self.it_fields.tauc_rec)
        self.vel_mes = retrieve_output()[2]*self.it_fields.mask
        self.it_fields.S_init = copy(self.it_fields.S_ref)

    def create_conf_file(self):
        filename = "icecap_conf.nc"
        nc = NC(filename, 'w', format="NETCDF3_CLASSIC")
        var = nc.createVariable("pism_overrides", 'i')

        attrs = {
         "geometry.update.use_basal_melt_rate": "no",
         "stress_balance.ssa.compute_surface_gradient_inward": "no",
         "flow_law.isothermal_Glen.ice_softness": self.it_parameters.A,
         "constants.ice.density": self.it_parameters.ice_density,
         "constants.sea_water.density": 1000.,
         "bootstrapping.defaults.geothermal_flux": 0.0,
         "stress_balance.ssa.Glen_exponent": 3.,
         "constants.standard_gravity": 9.81,
         "ocean.sub_shelf_heat_flux_into_ice": 0.0,
         "stress_balance.sia.bed_smoother.range": 0.0,
         }

        for name, value in attrs.items():
            var.setncattr(name, value)
        nc.close()
        
    def bootstrap(self):
        WRIT_FILE = self.file_locations.boot_in
        ##### define variables, set attributes, write data #####
        # format: ['units', 'long_name', 'standard_name', '_FillValue', array]

        vars = {'y':    ['m',
                         'y-coordinate in Cartesian system',
                         'projection_y_coordinate',
                         None,
                         self.it_fields.y],
                'x':    ['m',
                         'x-coordinate in Cartesian system',
                         'projection_x_coordinate',
                         None,
                         self.it_fields.x],
                'thk':  ['m',
                         'floating ice shelf thickness',
                         'land_ice_thickness',
                         None,
                         self.it_fields.S_rec - self.it_fields.B_rec],
                'topg': ['m',
                         'bedrock surface elevation',
                         'bedrock_altitude',
                         None,
                         self.it_fields.B_rec],
                'climatic_mass_balance': ['kg m-2 year-1',
                                          'mean annual net ice equivalent accumulation rate',
                                          'land_ice_surface_specific_mass_balance_flux',
                                          None,
                                          self.it_fields.smb],
                'ice_surface_temp': ['K',
                                     'annual mean air temperature at ice surface',
                                     'surface_temperature',
                                     273,
                                     self.it_parameters.ice_temp*np.ones_like(self.it_fields.S_ref)],
                'tauc': ['Pa',
                         'yield stress for basal till (plastic or pseudo-plastic model)',
                         'yield stress',
                         None,
                         self.it_fields.tauc_rec],
                }
        create_nc_input_data(vars, WRIT_FILE, int(np.shape(self.it_fields.x)[0]), int(np.shape(self.it_fields.y)[0]))

        original_stdout = sys.stdout # Save a reference to the original standard output
        with open(self.file_locations.boot_script, 'w') as f:
            sys.stdout = f
            create_script('forward', np.shape(self.it_fields.S_ref)[0], np.shape(self.it_fields.S_ref)[1])
            sys.stdout = original_stdout # Reset the standard output to its original value
            f.close()
                    
        cmd = ['chmod', '+x', self.file_locations.boot_script]
        subprocess.call(cmd)
        cmd = ['./{}'.format(self.file_locations.boot_script), '4', self.file_locations.boot_in, '1', self.file_locations.boot_out, '>', 'icecap_output_log.txt']
        subprocess.call(cmd)

    def create_it_script(self):
        original_stdout = sys.stdout # Save a reference to the original standard output
        with open(self.file_locations.it_script, 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            create_script('iteration', np.shape(self.it_fields.S_ref)[0], np.shape(self.it_fields.S_ref)[1])
            sys.stdout = original_stdout # Reset the standard output to its original value
            f.close()
        
    def update_nc(self):
        nc_updated = NC(self.file_locations.it_in, 'r+')
        nc_updated['topg'][0,:,:] = self.it_fields.B_rec
        nc_updated['thk'][0,:,:]=self.it_products.h_old
        nc_updated['tauc'][0,:,:]= self.it_fields.tauc_rec
        nc_updated.close()
        
    def short_forward(self):
        cmd = ['./{}'.format(self.file_locations.it_script), str(self.it_parameters.n_cores), self.file_locations.it_in, str(self.it_parameters.dt), self.file_locations.it_out]
        subprocess.call(cmd, stdout = open(self.file_locations.it_log, 'a'))
        
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
        self.it_fields.S_rec[np.logical_and(self.it_fields.mask == 1, self.it_products.h_rec>20)] = (self.it_fields.S_rec+self.it_parameters.delta_surf * self.it_products.misfit)[np.logical_and(self.it_fields.mask == 1, self.it_products.h_rec>20)]
        
    def interpolate_boundary(self):
        self.it_products.mask_iter = get_nc_data(self.file_locations.it_out, 'mask', 0)/2 #base buffer on ice mask produced by PISM --> faster than loop
        k = np.ones((self.it_parameters.bw, self.it_parameters.bw))
        buffer = ndimage.convolve(self.it_products.mask_iter, k)/(self.it_parameters.bw)**2 #smooth ice mask...
        self.it_products.criterion = np.logical_and(np.logical_and(buffer > 0, buffer != 1), self.it_fields.mask==1)
        self.it_fields.B_rec[self.it_products.criterion] = np.nan #...and take those values in the transition between ice and no ice;
        self.it_fields.B_rec = inpaint_nans(self.it_fields.B_rec)
        
    def mask_fields(self):
        self.it_fields.B_rec[self.it_fields.B_rec>=self.it_fields.S_rec] = self.it_fields.S_rec[self.it_fields.B_rec>=self.it_fields.S_rec] - 1
        self.it_fields.B_rec[self.it_fields.B_rec>=self.it_fields.S_ref] = self.it_fields.S_ref[self.it_fields.B_rec>=self.it_fields.S_ref] - 1
        self.it_fields.B_rec[self.it_fields.mask==0] = self.it_fields.S_ref[self.it_fields.mask==0]
        self.it_fields.S_rec[self.it_fields.mask==0] = self.it_fields.S_ref[self.it_fields.mask==0]
        
        
        self.it_fields.tauc_rec[self.it_fields.mask == 0] = self.it_fields.tauc_init[self.it_fields.mask == 0]
        
    def append_series(self):
        self.series.B_rec_all.append(self.it_fields.B_rec.copy())
        self.series.dh_all.append(self.it_products.dh_rec.copy())
        self.series.vel_all.append(self.it_products.vel_mod.copy())
        self.series.S_rec_all.append((self.it_fields.S_rec).copy())
        self.series.tauc_rec_all.append(self.it_fields.tauc_rec.copy())
        self.series.B_misfit_vs_iter.append(np.mean(abs(self.it_fields.B_rec - self.it_fields.B_ref)))
        self.series.dh_misfit_vs_iter.append(np.mean(abs(self.it_products.dh_rec[self.it_fields.mask==1] - self.it_fields.dh_ref[self.it_fields.mask==1])))
        self.series.vel_misfit_vs_iter.append(np.mean(abs(self.it_products.vel_mod[self.it_fields.mask==1]-self.it_fields.vel_mes[self.it_fields.mask==1])))
        self.series.tauc_misfit_vs_iter.append(np.mean(abs(self.it_fields.tauc_rec[self.it_fields.mask==1]-self.it_fields.tauc[self.it_fields.mask==1])))

        
    def update_tauc(self):
        self.it_products.vel_mismatch = np.maximum(np.minimum((np.maximum(self.it_products.vel_mod,0) - self.it_fields.vel_mes)/self.it_fields.vel_mes, .5), -.5)
        self.it_products.vel_mismatch[self.it_fields.mask==0]=np.nan
        self.it_products.vel_mismatch =  gauss_filter(self.it_products.vel_mismatch, .6, 2)
        self.it_products.vel_mismatch[np.isnan(self.it_products.vel_mismatch)]=0
        self.it_fields.tauc_rec = self.it_fields.tauc_rec+self.it_products.vel_mismatch* self.it_fields.tauc_rec * self.it_parameters.tauc_scale
        #self.it_fields.tauc_rec[np.logical_and(self.it_fields.mask==1, self.it_products.criterion)] = np.nan
        #self.it_fields.tauc_rec = inpaint_nans(self.it_fields.tauc_rec)    
        
    def iterate(self, input_data):
        self.copy_initial_state()
        self.create_conf_file()
        self.bootstrap()
        self.create_it_script()
        subprocess.call(['cp', self.file_locations.boot_out, self.file_locations.it_out])
        
        self.it_products.start_time = time.time()
        p=0
        while p < self.it_parameters.pmax:
            print(p)
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
            self.interpolate_boundary()
            self.mask_fields()
            self.append_series()
            if p>0 and p%self.it_parameters.p_friction == 0:
                self.update_tauc()
            p+=1
            if time.time() > self.it_products.start_time + self.it_parameters.max_time * 60 * 60:
                self.warnings.append('run did not finish in designated max time')
                break
