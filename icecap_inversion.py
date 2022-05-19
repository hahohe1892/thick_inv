import PISM
import numpy as np
from bed_inversion import *
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import subprocess
from netCDF4 import Dataset as NC
from icecap_get_reference_velocity import *

options = {
    "-Mz": 30,
    "-Lz": 5000,
    #"-z_spacing": "equal",
    "-surface" : "given",
    "-atmosphere.given.file": "input.nc",
    "-surface.given.file": "input.nc",
    "-ocean.given.file": "input.nc",
     "-i": "input.nc",
    "-bootstrap": "",
    "-energy": "none",
    "-sia_flow_law": "isothermal_glen",
    "-ssa_flow_law": "isothermal_glen",
    "-stress_balance": "ssa+sia",
    "-yield_stress": "constant",
    "-pseudo_plastic": "",
    "-pseudo_plastic_q": 0.333333,
    "-pseudo_plastic_uthreshold": 3.1556926e7,
    "-yield_stress": "constant",
    "geometry.update.use_basal_melt_rate": "no",
     "stress_balance.ssa.compute_surface_gradient_inward": "no",
     "flow_law.isothermal_Glen.ice_softness": 1.2597213016951452e-24,
     "constants.ice.density": 900.,
   "constants.sea_water.density": 1000.,
     "bootstrapping.defaults.geothermal_flux": 0.0,
     "stress_balance.ssa.Glen_exponent": 3.,
     "constants.standard_gravity": 9.81,
     "ocean.sub_shelf_heat_flux_into_ice": 0.0,
     "stress_balance.sia.bed_smoother.range": 0.0,
    "-o": "test.nc",
    "sea_level.constant.value": -1e4,
    "-bed_def": "iso",
    "-bed_deformation.mantel_density": 1e20
}


cmd = ['cp', 'icecap_initial_setup.nc', 'input.nc']
subprocess.call(cmd)

inversion_in = NC('input.nc', 'r+')
inversion_in['topg'][:,:] = np.zeros((51,51))
inversion_in['usurf'][:,:] = get_nc_data('ice_build_output.nc', 'usurf', 0)
inversion_in['mask'][:,:] = get_nc_data('ice_build_output.nc', 'mask', 0)
inversion_in['thk'][:,:] = get_nc_data('ice_build_output.nc', 'usurf', 0)
inversion_in['velsurf_mag'][:,:] = get_nc_data('ice_build_output.nc', 'velsurf_mag', 0)
inversion_in.close()

pism = create_pism("input.nc", options)

tauc_rec = np.array(pism.basal_yield_stress_model().basal_material_yield_stress().local_part(),copy=True)/1000

dh_ref = np.zeros_like(tauc_rec)
B_rec = np.zeros_like(tauc_rec)
S_rec = np.array(pism.geometry().ice_surface_elevation.local_part(), copy=True)
mask = np.array(pism.geometry().cell_type.local_part(), copy=True)/2
vel_ref = get_reference_velocity()*mask

B_rec_old = np.copy(B_rec)
S_rec_old = np.copy(S_rec)

dt = .1
beta = 1
bw = 5
pmax = 1000
p_friction = 500

B_rec_all = []
misfit_all = []

for p in range(pmax):

    if p>0 and p%p_friction == 0:
        update_friction = 'yes'
    else:
        update_friction = 'no'
        
    B_rec, S_rec, tauc_rec, misfit = iteration(pism, B_rec, S_rec, tauc_rec, mask, dh_ref, vel_ref, dt, beta, bw, update_friction)
    B_rec_all.append(np.copy(B_rec))
    misfit_all.append(misfit)

print(B_rec.shape)
pism.save_results()
dh_misfit_vs_iter = [np.nanmean(abs(i[mask==1])) for i in misfit_all]

"""
colormap = plt.cm.viridis
colors = [colormap(i) for i in np.linspace(0,1,len(B_rec_all))]

fig, ax = plt.subplots(1,3, figsize=(15,4))
for i in range(len(B_rec_all)):
    lines = ax[0].plot(range(B_rec_all[i].shape[0]), B_rec_all[i][21,:], color = colors[i])

field = ax[1].pcolor(B_rec)
fig.colorbar(field, ax = ax[1])

lines1 = ax[2].plot(dh_misfit_vs_iter)
ax[0].set_xlabel('x-coordinate')
ax[0].set_ylabel('recovered bed elevation[m]')
plt.show()
"""
