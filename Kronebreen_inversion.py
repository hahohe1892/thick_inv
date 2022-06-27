import PISM
import numpy as np
from bed_inversion import *
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import subprocess
from netCDF4 import Dataset as NC


options = {
    "-Mz": 50,
    "-Lz": 1500,
    "-Lbz": 1,
    "-surface" : "given",
    "-surface.given.file": "Kronebreen_input.nc",
    "-i": "Kronebreen_input.nc",
    "-bootstrap": "",
    "-energy": "none",
    "-sia_flow_law": "isothermal_glen",
    "-ssa_flow_law": "isothermal_glen",
    "-stress_balance": "ssa+sia",
    "-yield_stress": "constant",
    "-pseudo_plastic": "",
    "-pseudo_plastic_q": 0.2,
    "-pseudo_plastic_uthreshold": 3.1556926e7,
    "-geometry.update.use_basal_melt_rate": "no",
    "-stress_balance.ssa.compute_surface_gradient_inward": "no",
    "-flow_law.isothermal_Glen.ice_softness": 3.9565534675428266e-24,
    "-constants.ice.density": 900.,
    "-constants.sea_water.density": 1000.,
    "-bootstrapping.defaults.geothermal_flux": 0.0,
    "-stress_balance.ssa.Glen_exponent": 3.,
    "-constants.standard_gravity": 9.81,
    "-ocean.sub_shelf_heat_flux_into_ice": 0.0,
    "-stress_balance.sia.bed_smoother.range": 0.0,
    "-o": "Kronebreen_output.nc",
    "-sea_level.constant.value": -1e20,
    "-time_stepping.assume_bed_elevation_changed": "true"
    }


cmd = ['cp', 'Kronebreen_initial_setup.nc', 'Kronebreen_input.nc']
subprocess.call(cmd)

pism = create_pism("Kronebreen_input.nc", options)

dh_ref = read_variable( pism.grid(), "Kronebreen_input.nc", 'dhdt', 'm year-1')
mask = read_variable( pism.grid(), "Kronebreen_input.nc", 'mask', '')
vel_ref = read_variable( pism.grid(), "Kronebreen_input.nc", 'velsurf_mag', 'm year-1')
contact_zone = read_variable( pism.grid(), "Kronebreen_input.nc", 'contact_zone', '')
ocean_mask = read_variable( pism.grid(), "Kronebreen_input.nc", 'ocean_mask', '')
    
tauc_rec = np.array(pism.basal_yield_stress_model().basal_material_yield_stress().local_part(),copy=True)

B_rec = np.array(pism.geometry().bed_elevation.local_part(), copy=True)
S_rec = np.array(pism.geometry().ice_surface_elevation.local_part(), copy=True)

# set inversion paramters
dt = .1
beta = .5
bw = 0
pmax = 100
p_friction = 1000
max_steps_PISM = 20
res = 250
A = 3.9565534675428266e-24

B_init = np.copy(B_rec)
B_rec_all = []
misfit_all = []

# do the inversion
for p in range(pmax):

    if p>0 and p%p_friction == 0:
        update_friction = 'yes'
    else:
        update_friction = 'no'
        
    B_rec, S_rec, tauc_rec, misfit = iteration(pism,
                                               B_rec, S_rec, tauc_rec, mask, dh_ref, vel_ref,
                                               dt = dt,
                                               beta = beta,
                                               bw = bw,
                                               update_friction = update_friction,
                                               res=res,
                                               A=A,
                                               max_steps_PISM = max_steps_PISM,
                                               treat_ocean_boundary = 'no',
                                               correct_diffusivity = 'yes',
                                               contact_zone = contact_zone,
                                               ocean_mask = ocean_mask)
    B_rec_all.append(np.copy(B_rec))
    misfit_all.append(misfit)

pism.save_results()
dh_misfit_vs_iter = [np.nanmean(abs(i[mask==1])) for i in misfit_all]
