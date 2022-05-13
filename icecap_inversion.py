import PISM
import numpy as np
from bed_inversion import *
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import subprocess
from netCDF4 import Dataset as NC

cmd = ['cp', 'ice_build_output.nc', 'ice_build_output_mod.nc']
subprocess.call(cmd)

nc_updated = NC('ice_build_output_mod.nc', 'r+')
nc_updated['topg'][0,:,:]=np.zeros((51, 51))
nc_updated['thk'][0,:,:]=nc_updated['usurf'][0,:,:]
nc_updated.close()


options = {
    "-Mz": 30,
    "-Lz": 5000,
    #"-z_spacing": "equal",
    "-surface" : "given",
    "-surface_given_file": "ice_build_output_mod.nc",
    "-i": "ice_build_output_mod.nc",
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
    "-o": "test.nc"
}

pism = create_pism("ice_build_output_mod.nc", options)

H   = np.array(pism.geometry().ice_thickness.local_part(), copy=True)
bed = np.array(pism.geometry().bed_elevation.local_part(), copy=True)

usurf = np.array(pism.geometry().ice_surface_elevation.local_part(), copy=True)
mask = np.array(pism.geometry().cell_type.local_part(), copy=True)/2
dh_ref = np.zeros_like(usurf)

tauc = np.array(pism.basal_yield_stress_model().basal_material_yield_stress().local_part(),copy=True)

B_rec = np.copy(bed)
S_rec = np.copy(usurf)
B_rec_all = []
S_rec_all = []

for i in range(100):
    B_rec, S_rec = iteration(pism, B_rec, S_rec, tauc, mask, dh_ref, .1, 1, 3)
    B_rec_all.append(B_rec)
    S_rec_all.append(S_rec)

# plot cross-section through bed and surface
colormap = plt.cm.viridis
colors = [colormap(i) for i in np.linspace(0,1,len(B_rec_all))]
fig, ax = plt.subplots(1,2, figsize=(15,4))
for i in range(len(B_rec_all)):
    lines = ax[0].plot(range(B_rec_all[i].shape[0]), B_rec_all[i][21,:], color = colors[i])
    lines1 = ax[1].plot(range(S_rec_all[i].shape[0]), S_rec_all[i][21,:], color = colors[i])
ax[0].set_xlabel('x-coordinate')
ax[1].set_xlabel('x-coordinate')
ax[0].set_ylabel('recovered bed elevation [m]')
ax[1].set_ylabel('recovered surface elevation [m]')
fig.show()
