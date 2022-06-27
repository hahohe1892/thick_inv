import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import rasterio
from rasterio import mask as msk
import geopandas as gpd
from copy import deepcopy
from copy import copy
import pandas as pd
from scipy import ndimage
from scipy import spatial
from scipy import interpolate
from scipy.ndimage import zoom
from netCDF4 import Dataset as NC
from funcs import *

class input_data():
    
    def __init__(self):
            self.x, self.y = np.meshgrid(1,1)
            self.smb = np.zeros_like(self.x, dtype='float')
            self.dem = np.zeros_like(self.x, dtype='float')
            self.NPI_DEM = np.zeros_like(self.x, dtype='float')            
            self.dhdt = np.zeros_like(self.x, dtype='float')
            self.mask = np.zeros_like(self.x, dtype='float')
            self.vel_Jack = np.zeros_like(self.x, dtype='float')
            self.vx_Jack = np.zeros_like(self.x, dtype='float')
            self.vy_Jack = np.zeros_like(self.x, dtype='float')
    
    def check_shape(self):
        check_shape(self.x, [self.dem, self.smb, self.dhdt, self.mask, self.vel_Jack, self.vx_Jack, self.vy_Jack])

    def set_xy(self, matrix):
        self.x, self.y = np.meshgrid(matrix[0],np.arange(np.min(matrix[1]), np.max(matrix[1])+10050, 50))
   
    def reset_shape(self):
        self.dem = np.zeros_like(self.x, dtype='float')
        self.NPI_DEM = np.zeros_like(self.x, dtype='float')
        self.smb = np.zeros_like(self.x, dtype='float')
        self.dhdt = np.zeros_like(self.x, dtype='float')
        self.mask = np.zeros_like(self.x, dtype='float')
        self.vel_Jack = np.zeros_like(self.x, dtype='float')
        self.vx_Jack = np.zeros_like(self.x, dtype='float')
        self.vy_Jack = np.zeros_like(self.x, dtype='float')
                
    def import_NPI_DEM(self, path):
        self.NPI_DEM_o = rasterio.open(path)
        self.window = self.NPI_DEM_o.window(np.min(self.x), np.min(self.y), np.max(self.x), np.max(self.y))
        self.NPI_DEM[1:,1:] = np.flip(self.NPI_DEM_o.read(1, window = self.window), axis = 0) 
        
    def get_vel(self, matrix):
        
        self.vx_Jack = np.zeros_like(self.x, dtype='float')
        self.vx_Jack[:np.shape(self.x)[0]-int(10000/50),:] = matrix[6]
        self.vx_Jack[np.isnan(self.vx_Jack)] = 0
        
        self.vy_Jack = np.zeros_like(self.x, dtype='float')
        self.vy_Jack[:np.shape(self.x)[0]-int(10000/50),:] = matrix[7]
        self.vy_Jack[np.isnan(self.vy_Jack)] = 0
        
        self.vel_Jack = np.zeros_like(self.x, dtype='float')
        self.vel_Jack[:np.shape(self.x)[0]-int(10000/50),:] = matrix[8]*365*3.8
        self.vel_Jack[np.isnan(self.vel_Jack)] = 0
        
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

        #self.mask_Kr = np.zeros_like(self.x, dtype='float')
        #self.mask_Kr[(np.shape(self.mask_Kr)[0]-np.shape(NPI_mask_Kr)[0]+shift_y_Kr):(np.shape(self.mask_Kr)[0]+shift_y_Kr), shift_x_Kr:(shift_x_Kr+np.shape(NPI_mask_Kr)[1])]=NPI_mask_Kr

    def get_dhdt(self, matrix):
        self.dhdt0 = np.zeros_like(self.x, dtype='float')*np.nan
        self.dhdt1 = np.zeros_like(self.x, dtype='float')*np.nan
        self.dhdt0[:np.shape(self.x)[0]-int(10000/50),:] = matrix[4]
        self.dhdt1[:np.shape(self.x)[0]-int(10000/50),:] = matrix[5]
    
        self.mask_Kr = np.ones_like(self.x)
        self.mask_Kr[np.isnan(self.dhdt1)] = 0
        self.mask_Kr[0:10,:]=0

        
    def get_mask(self):
        #self.retreat_mask = np.zeros_like(self.x, dtype='float')
        #self.retreat_mask[np.logical_and(self.NPI_DEM<100, np.logical_and(self.dhdt0>1, -1*self.mask_Kr+1))] = 1
        #self.retreat_mask[self.retreat_mask+self.mask_Kr==2]=0
        #self.retreat_mask[np.logical_and(np.isnan(self.dhdt1), np.isnan(self.dhdt0)==False)]=1
        
        self.mask = copy(self.mask_Kr)
        self.mask[self.mask>0] = 1
        
        #self.NPI_DEM[self.retreat_mask==1] = 0

    def clean_dhdt(self):
        
        ## set ocean (i.e. what is not land or glacier) to negative value
        self.ocean_mask = np.zeros_like(self.x)
        #self.ocean_mask[self.NPI_DEM<90]=1
        self.ocean_mask[np.logical_and(self.mask==0, self.NPI_DEM<150)]=1  #this should reflect area where the glacier has retreated
        #self.ocean_mask[np.logical_and(np.isnan(self.dhdt0), self.NPI_DEM<5)]=1
        self.ocean_mask[:,120:-1]=0
        self.ocean_mask[:10,:]=0
        self.ocean_mask[:,:10]=0
        self.ocean_mask[self.mask==1]=0

        self.dhdt0[np.isnan(self.dhdt0)] = self.NPI_DEM[np.isnan(self.dhdt0)]
        self.dhdt1[np.isnan(self.dhdt1)] = self.NPI_DEM[np.isnan(self.dhdt1)]

        self.dhdt0[self.mask_Kr==0] = self.NPI_DEM[self.mask_Kr==0]
        self.dhdt1[self.mask_Kr==0] = self.NPI_DEM[self.mask_Kr==0]
        
        self.dhdt = (self.dhdt1 - self.dhdt0)/6
        self.dhdt[np.isnan(self.dhdt)] = 0
        self.dem = copy(self.dhdt1)#(self.dhdt0+self.dhdt1)/2
        self.dem[self.ocean_mask==1]=-100
        self.dhdt[self.ocean_mask==1]=0

        
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
        
        smb_net_1420 = np.array(smb_net_df.loc[1:,12:18])/100*(10/9)   #convert to m.ice.eq.
        smb_net_1420 = np.nanmean(smb_net_1420, axis=1)
        
        ### interpolate smb with elevation ###
        poly = np.poly1d(np.polyfit(smb_z,smb_net_1420,1))
        self.smb = poly(self.dem)
        self.smb[self.smb<-2] = -2
        self.smb[self.smb>1] = 1
        
    def get_vel_stake(self, path):
        ### velocity from Jack ###
        vel_xyz_df = pd.read_excel(path, 1, header=None)
        vel_x = np.array(vel_xyz_df.loc[:,1])
        vel_y = np.array(vel_xyz_df.loc[:,2])
        vel_z = np.array(vel_xyz_df.loc[:,3])
        
        vel_df = pd.read_excel (path, 0, header=None)
        self.vel_stake = np.nanmean(np.array(vel_df.loc[3:,range(20,30,2)]), axis=1)

    def get_vel_Adrian(self):
        path = './kronebreen/AL_vels_HDF.mat'
        Adrian_vel_mat = scipy.io.loadmat(path)['AL_vels_HDF'][0,0]
        self.vel_Adrian = 365*(scipy.interpolate.griddata(((Adrian_vel_mat[0]).flatten(), (Adrian_vel_mat[1]).flatten()), Adrian_vel_mat[2].flatten(), ((self.x).flatten(), (self.y).flatten()))).reshape(np.shape(self.x))
        self.vx_Adrian = 365*(scipy.interpolate.griddata(((Adrian_vel_mat[0]).flatten(), (Adrian_vel_mat[1]).flatten()), Adrian_vel_mat[3].flatten(), ((self.x).flatten(), (self.y).flatten()))).reshape(np.shape(self.x))
        self.vy_Adrian = 365*(scipy.interpolate.griddata(((Adrian_vel_mat[0]).flatten(), (Adrian_vel_mat[1]).flatten()), Adrian_vel_mat[4].flatten(), ((self.x).flatten(), (self.y).flatten()))).reshape(np.shape(self.x))

        self.vel_Adrian[np.isnan(self.vel_Adrian)] = self.vel_Jack[np.isnan(self.vel_Adrian)]
        self.vx_Adrian[np.isnan(self.vel_Adrian)] = self.vx_Jack[np.isnan(self.vel_Adrian)]
        self.vy_Adrian[np.isnan(self.vel_Adrian)] = self.vy_Jack[np.isnan(self.vel_Adrian)]

    def get_data_Millan(self):
        thk_o = rasterio.open('./kronebreen/THICKNESS_RGI-7.1_2021July09.tif')  
        window_thk = (thk_o.window(np.min(self.x), np.min(self.y), np.max(self.x), np.max(self.y),1))
        self.thk_Millan_in = np.flip(thk_o.read(1, window=window_thk), axis = 0)

        vel_o = rasterio.open('./kronebreen/V_RGI-7.1_2021July01.tif')
        window_vel = (vel_o.window(np.min(self.x), np.min(self.y), np.max(self.x), np.max(self.y),1))
        self.vel_Millan_in = np.flip(vel_o.read(1, window=window_vel), axis = 0)
        self.vel_Millan_in[np.isnan(self.vel_Millan_in)] = 0
        '''
        destination = np.zeros((np.shape(self.x)[0], np.shape(self.x)[1]+3))
        self.vel_Millan = rasterio.warp.reproject(
            vel_in,
            destination,
            src_transform=vel_o.transform,
            src_crs=vel_o.crs,
            dst_transform=self.NPI_DEM_o.transform,
            dst_crs=self.NPI_DEM_o.crs,
            resampling=rasterio.warp.Resampling.nearest)[0]
        
        self.thk_Millan = rasterio.warp.reproject(
            thk_in,
            destination,
            src_transform=thk_o.transform,
            src_crs=thk_o.crs,
            dst_transform=self.NPI_DEM_o.transform,
            dst_crs=self.NPI_DEM_o.crs,
            resampling=rasterio.warp.Resampling.nearest)[0]
        '''
    def resample_input(self):
        self.data_res = 50
        self.res = 250
        self.resample = self.data_res/self.res
        
        self.dhdt0 = zoom(self.dhdt0, self.resample)
        self.dhdt1 = zoom(self.dhdt1, self.resample)
        self.dhdt = zoom(self.dhdt, self.resample)
        self.smb = zoom(self.smb, self.resample)
        self.dem = zoom(self.dem, self.resample)
        self.NPI_DEM = zoom(self.NPI_DEM, self.resample)
        self.vel_Jack = zoom(self.vel_Jack, self.resample)
        self.vx_Jack = zoom(self.vx_Jack, self.resample)
        self.vy_Jack = zoom(self.vy_Jack, self.resample)
        self.vel_Adrian = zoom(self.vel_Adrian, self.resample)
        self.vx_Adrian = zoom(self.vx_Adrian, self.resample)
        self.vy_Adrian = zoom(self.vy_Adrian, self.resample)
        self.vel_Millan = np.zeros_like(self.NPI_DEM)
        self.vel_Millan[:,1:] = zoom(self.vel_Millan_in, self.resample)
        self.thk_Millan = np.zeros_like(self.NPI_DEM)
        self.thk_Millan[:,1:] = zoom(self.thk_Millan_in, self.resample)
        
        self.mask = np.around(zoom(self.mask, self.resample), 0)
        self.mask_Kr = np.around(zoom(self.mask_Kr, self.resample), 0)
        self.mask_Ko = np.around(zoom(self.mask_Ko, self.resample), 0)
        self.ocean_mask = np.around(zoom(self.ocean_mask, self.resample), 0)
              
        self.x = zoom(self.x, self.resample)
        self.y = zoom(self.y, self.resample)

        self.contact_zone = np.zeros_like(self.x)
        for i in range(np.shape(self.contact_zone)[1]):
            for j in range(np.shape(self.contact_zone)[0]):
                if self.ocean_mask[j,i]==0:
                    continue
                x, y = np.ogrid[:np.shape(self.x)[0], :np.shape(self.x)[1]]
                circle = (y - i) ** 2 + (x - j) ** 2 < 1.5
                self.contact_zone[np.logical_and(self.mask==1, circle)]=1     

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

    def fill_in_vel(self):
        Jack_missing = np.where(np.logical_and(self.itslive_vel>300, self.vel_Jack<30))
        self.vel_Jack[Jack_missing]=self.itslive_vel[Jack_missing]

        Adrian_missing = np.where(np.logical_and(self.vel_Jack>=300, self.vel_Adrian<30))
        self.vel_Adrian[Adrian_missing]=self.vel_Jack[Adrian_missing]

        Millan_missing = np.where(np.logical_and(self.vel_Jack>=500, self.vel_Millan<300))
        self.vel_Millan[Millan_missing]=self.vel_Jack[Millan_missing]
        
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

    def get_boundary(self):
        self.boundary = np.zeros_like(self.x)
        for i in range(np.shape(self.boundary)[1]):
            for j in range(np.shape(self.boundary)[0]):
                if self.mask[j,i]==1:
                    continue
                x, y = np.ogrid[:np.shape(self.x)[0], :np.shape(self.x)[1]]
                circle = (y - i) ** 2 + (x - j) ** 2 < 1.5
                self.boundary[np.logical_and(self.mask==1, circle)]=1
                
    def set_parameters(self, ice_temp=273, ice_density = 900., secpera = 31556926., g = 9.81):
        self.ice_temp = ice_temp
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

        self.get_outlines('./kronebreen/07_rgi60_Svalbard.shp')

        self.get_dhdt(DEM_VEL_mat)

        self.get_mask()

        self.clean_dhdt()

        self.set_dhdt_Ko()

        self.set_smb('./kronebreen/HDF_mass_balance.xlsx')

        self.get_vel_stake('./kronebreen/HDF_stake_velocities.xlsx')

        self.get_vel_Adrian()

        self.get_data_Millan()

        self.resample_input()

        self.get_itslive_vel(['./kronebreen/vel_ITSLIVE_resample.tif', "./kronebreen/vx_ITSLIVE_proj.tif", "./kronebreen/vy_ITSLIVE_proj.tif"])

        self.fill_in_vel()

        self.filter_dhdt()

        self.get_boundary()

        self.set_parameters()


def initial_bed_guess(observations, n = 20):
    dH = (np.nanmax(observations.dem[observations.mask==1])-np.nanmin(observations.dem[observations.mask==1]))/1000 #in km
    
    tau = 0.005+1.598*dH-0.435*dH**2  #Haeberli and Hoelzle
    
    S_stag = stagger(observations.dem)
    dhdx = x_deriv(S_stag, observations.res)
    dhdy = y_deriv(S_stag, observations.res)
    observations.dem_slope = np.sqrt(dhdx**2+dhdy**2)
    
    observations.dem_slope[observations.dem_slope<0.015] = 0.015
    observations.dem_sin_slope = np.sin(np.arctan(observations.dem_slope))
    H = ((tau*.5)*1e5)/(observations.dem_sin_slope*observations.g*observations.ice_density)
    H_smooth = deepcopy(H)
    sin_slope_smooth = np.copy(observations.dem_sin_slope)
    for i in range(n):
        sin_slope_smooth = smooth_stress_coupling(H_smooth, observations.dem_sin_slope, observations.mask, 4, 8,2, observations.res)
        H_smooth =(tau*.5)*1e5/(sin_slope_smooth*observations.g*observations.ice_density)
    
    observations.initial_thickness = H_smooth*(-1*observations.ocean_mask+1)
    
    return observations

    
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


def smooth_SandB(surf, bed, mask, surface_smooth=(1,3), bed_smooth=(.6, .3)):
    surf_in = np.copy(surf)
    surf[mask==0] = np.nan
    surf = gauss_filter(surf, surface_smooth[0], surface_smooth[1])
    surf[mask==0] = surf_in[mask==0]

    bed[mask==0] = np.nan
    bed = gauss_filter(bed, bed_smooth[0], bed_smooth[1])
    bed[mask == 0] = surf_in[mask==0]
    bed[bed > surf] = surf[bed > surf]

    return surf, bed

def set_topography():
    observations = input_data()
    observations.build_input()
    print('building input data done!')

    ny = observations.dem.shape[0]
    nx = observations.dem.shape[1]

    x_in = observations.x[0]
    y_in = observations.y[:,1]

    initial_thickness = initial_bed_guess(observations, 20).initial_thickness
    initial_bed = observations.dem - initial_thickness

    initial_surf, initial_bed = smooth_SandB(observations.dem, initial_bed, observations.mask)
    initial_bed = correct_high_diffusivity(initial_surf, initial_bed, dt=0.1, max_steps_PISM = 50, res=observations.res, A=observations.A)
    initial_thickness = initial_surf - initial_bed
    
    tauc = (500+initial_bed)*1e3
    tauc[observations.ocean_mask==1]=np.median(tauc[observations.mask==1])
    
    setup = {'topg': initial_bed,
             'dhdt': observations.dhdt,
             'usurf': initial_surf,
             'mask': observations.mask,
             'tauc': tauc,
             'velsurf_mag': observations.vel_Millan,
             'ice_surface_temp': np.ones_like(observations.dem)*270,
             'smb': observations.smb,
             'thk': initial_thickness,
             'contact_zone': observations.contact_zone,
             'ocean_mask': observations.ocean_mask,
             'x': x_in,
             'y': y_in}

    return setup


def write_setup_to_nc(setup, output_file):
    """ takes dictionary with required input fields 
    (i.e. topg, tauc, ice_surface_temp, smb, thk, x, y)
    and writes an nc file that can be used to initialize PISM """
    
    vars = {'y':    ['m',
                 'y-coordinate in Cartesian system',
                 'projection_y_coordinate',
                 None,
                setup['y']],
        'x':    ['m',
                 'x-coordinate in Cartesian system',
                 'projection_x_coordinate',
                 None,
                 setup['x']],
        'thk':  ['m',
                 'floating ice shelf thickness',
                 'land_ice_thickness',
                 None,
                 setup['thk']],
        'topg': ['m',
                 'bedrock surface elevation',
                 'bedrock_altitude',
                 None,
                 setup['topg']],
        'ice_surface_temp': ['K',
                             'annual mean air temperature at ice surface',
                             'surface_temperature',
                             None,
                             setup['ice_surface_temp']],
        'climatic_mass_balance': ['kg m-2 year-1',
                                  'mean annual net ice equivalent accumulation rate',
                                  'land_ice_surface_specific_mass_balance_flux',
                                  None,
                                  setup['smb'] * 900],
        'tauc': ['Pa',
                 'yield stress for basal till (plastic or pseudo-plastic model)',
                 'yield stress',
                 None,
                 setup['tauc']],
        'mask': ['',
                 'ice-type (ice-free/grounded/floating/ocean) integer mask',
                 'mask',
                 None,
                 setup['mask']],
        'usurf': ['m',
                 'ice top surface elevation',
                 'surface_altitude',
                 None,
                  setup['usurf']],
        'velsurf_mag': ['m year-1',
                        'magnitude of horizontal velocity of ice at ice surface',
                        'surface velocity',
                        None,
                        setup['velsurf_mag']],
        'dhdt': ['m year-1',
                 'surface elevation change in meters ice per year',
                 'dhdt',
                 None,
                 setup['dhdt']],
        'contact_zone': ['',
                 'ocean area that is in contact with the ice fron',
                 'ocean contact',
                 None,
                setup['contact_zone']],
        'ocean_mask': ['',
                 'part of domain that is ocean',
                 'ocean',
                 None,
                 setup['ocean_mask']]
            
        }
    
    ncfile = NC(output_file, 'w', format='NETCDF3_CLASSIC')
    xdim = ncfile.createDimension('x', int(setup['x'].shape[0]))
    ydim = ncfile.createDimension('y', int(setup['y'].shape[0]))
    
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
    print("NetCDF file ", output_file, " created")
    print('')


if __name__ == "__main__":
        
    setup_file = "Kronebreen_initial_setup.nc"
        
    setup = set_topography()
    write_setup_to_nc(setup, setup_file)
