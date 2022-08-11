 #!/bin/bash

for S_ref_rand in 5 10 15 25 50
do
    mpiexec -n 1 python3 icecap_inversion.py -S_ref_rand $S_ref_rand
done


: <<'END'

import numpy as np
from funcs import *
from bed_inversion import *
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
bed_deviations = []
bed_deviations_tf = []
taucs = []
vels = []
vols = []
vols_tf = []
fluxes = []
beds = []
surfs = []
surfs_in =[]
rsmes = []
true_bed = get_nc_data('ice_build_output.nc', 'topg', -1)
true_surf = get_nc_data('ice_build_output.nc', 'usurf', -1)
true_vol = np.sum(true_surf - true_bed)
true_tauc = get_nc_data('ice_build_output.nc', 'tauc', -1)
no_sliding_surf = get_nc_data('icecap_output_no_friction_update.nc', 'usurf', -1)
no_sliding_bed = get_nc_data('icecap_output_no_friction_update.nc', 'topg', -1)
no_sliding_vol = np.sum(no_sliding_surf - no_sliding_bed)

reference_bed = get_nc_data('icecap_output_mb_1.0.nc', 'topg', -1)
reference_surf = get_nc_data('icecap_output_mb_1.0.nc', 'usurf', -1)
reference_vol = np.sum(reference_surf - reference_bed)
reference_tauc = get_nc_data('icecap_output_mb_1.0.nc', 'tauc', -1)
reference_vel  = get_nc_data('icecap_output_mb_1.0.nc', 'velbar_mag', -1)
reference_flux = get_nc_data('icecap_output_mb_1.0.nc', 'flux_mag', -1)
mask = get_nc_data('ice_build_output.nc', 'mask', -1).data/2
xs = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]
for i in xs:
    bed = get_nc_data('icecap_output_mb_{}.nc'.format(i), 'topg', -1)
    bed_tf = get_nc_data('icecap_output_mb_{}_true_friction.nc'.format(i), 'topg', -1)
    beds.append(bed)
    surf = get_nc_data('icecap_output_mb_{}.nc'.format(i), 'usurf', -1)
    surfs.append(surf)
    surf_tf =  get_nc_data('icecap_output_mb_{}_true_friction.nc'.format(i), 'usurf', -1)
    surfs_in.append(np.maximum(true_surf + np.random.normal(0, x, np.shape(true_surf)), true_bed))
    vols.append(np.sum(surf-bed))
    vols_tf.append(np.sum(surf_tf-bed_tf))
    taucs.append(np.nanmedian(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'tauc', 0)[mask==1]))
    vels.append(np.nanmean(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'velbar_mag', 0)[mask==1]))
    fluxes.append(np.sum(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'flux_mag', 0)[mask==1]))
    bed_deviations.append(rmse(bed[mask==1], true_bed[mask==1]))
    bed_deviations_tf.append(rmse(bed_tf[mask==1], true_bed[mask==1]))

taucs.append(np.median(reference_tauc[mask==1]))
vels.append(np.nanmean(reference_vel[mask==1]))
fluxes.append(np.sum(reference_flux))
vols_tf.append(reference_vol)
xs.append(1)
vols.append(reference_vol)
bed_deviations.append(rmse(reference_bed[mask==1], true_bed[mask==1]))
bed_deviations_tf.append(rmse(reference_bed[mask==1], true_bed[mask==1]))

fig, ax = plt.subplots()
line1 = ax.plot(np.sort(xs), np.sort(np.array(xs)**(1/5)), '--', c = 'black', label='5th root of mass balance errors', alpha = .5)
points1 = ax.scatter(xs,vols_tf/reference_vol, c = bed_deviations_tf, label='true friction field prescribed', cmap='plasma', marker = 'v', vmin = 0, vmax = np.max(bed_deviations))
points = ax.scatter(xs, vols/true_vol, label = 'applying friction updates', c=bed_deviations, cmap='plasma')
ax.set_xlabel('mass balance / true mass balance')
ax.set_ylabel('modelled ice volume / true ice volume')
cbar = fig.colorbar(points, ax = ax)
cbar.ax.set_ylabel('RMSE')
ax1 = plt.axes([0,0,1,1])
ip1 = InsetPosition(ax, [0.1,0.65,0.3,0.3])
ax1.set_axes_locator(ip1)
ax1.pcolor(beds[0], vmin = -500, vmax = 1000)
ax1.axis('off')
ax2 = plt.axes([0,0,.99,1])
ip2 = InsetPosition(ax, [0.65,0.2,0.3,0.3])
ax2.set_axes_locator(ip2)
ax2.pcolor(beds[-1], vmin = -500, vmax = 1000)
ax2.axis('off')
ax.legend()
leg = ax.get_legend()
leg.legendHandles[1].set_color('black')
leg.legendHandles[2].set_color('black')
plt.savefig('./figures/mass_balance_errors_v1.1.png')
plt.show()

model = LinearRegression().fit(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations)))


for surf in surfs:
    surf_deviations.append(np.mean(abs(surf[mask==1] - true_surf[mask==1])))



fig, ax = plot_list(beds, 3,2, vmin = -500, vmax = 1000)
ax[2,1].pcolor(true_bed, vmin = -500, vmax = 1000)
plt.show()

fig,ax = plt.subplots()
line = ax.plot(range(true_bed.shape[1]), true_bed[27,:])
line = ax.plot(range(true_bed.shape[1]), true_surf[27,:])
for i in range(len(beds)):
    line = ax.plot(range(true_bed.shape[1]), beds[i][27,:], c='r')
    line = ax.plot(range(true_bed.shape[1]), surfs_in[i][27,:], c='r')    

plt.show()
END
