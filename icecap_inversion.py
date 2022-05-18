import PISM
import numpy as np
from bed_inversion import *
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import subprocess
from netCDF4 import Dataset as NC

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
    "sea_level.constant.value": -1e4
}


cmd = ['cp', 'icecap_initial_setup.nc', 'input.nc']
subprocess.call(cmd)

inversion_in = NC('input.nc', 'r+')
inversion_in['topg'][:,:] = np.zeros((51,51))
inversion_in.close()

pism = create_pism("input.nc", options)

tauc = np.array(pism.basal_yield_stress_model().basal_material_yield_stress().local_part(),copy=True)

dh_ref = np.zeros_like(tauc)
B_rec = np.zeros((55,55))
S_rec = np.zeros_like(B_rec)
S_rec[2:-2,2:-2] = get_nc_data('ice_build_output.nc', 'usurf', 0)

mask = np.zeros_like(B_rec)
mask[2:-2,2:-2] = get_nc_data('ice_build_output.nc', 'mask', 0)

B_rec_old = np.copy(B_rec)
S_rec_old = np.copy(S_rec)

dt = .1
beta = 1
bw = 3

B_rec_all = []
B_rec_old_all = []

for i in range(100):
    # new (i.e. broken) implementation
    B_rec, S_rec = iteration(pism, B_rec, S_rec, tauc, mask, dh_ref, dt, beta, bw)
    # old implementation (i.e. write to file, initialize PISM, read from file)
    B_rec_old, S_rec_old = iteration_old(B_rec_old, S_rec_old, tauc, mask, dh_ref, dt, beta, bw, options)
    B_rec_all.append(np.copy(B_rec))
    B_rec_old_all.append(np.copy(B_rec_old))

# plot cross-section through bed and surface
colormap = plt.cm.viridis
colors = [colormap(i) for i in np.linspace(0,1,len(B_rec_all))]
fig, ax = plt.subplots(1,2, figsize=(15,4))
for i in range(len(B_rec_all)):
    lines = ax[0].plot(range(B_rec_all[i].shape[0]), B_rec_all[i][21,:], color = colors[i])
    lines1 = ax[1].plot(range(B_rec_old_all[i].shape[0]), B_rec_old_all[i][21,:], color = colors[i])
ax[0].set_xlabel('x-coordinate')
ax[1].set_xlabel('x-coordinate')
ax[0].set_ylabel('recovered bed elevation without reinitialization[m]')
ax[1].set_ylabel('recovered surface elevation with reinitialization[m]')
fig.show()
