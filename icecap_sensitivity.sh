 #!/bin/bash

for theta in 0 0.015 0.05 0.075 0.1 0.5
do
    mpiexec -n 1 python3 icecap_inversion.py -theta $theta
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
    bed = get_nc_data('icecap_output_vel_{}.nc'.format(i), 'topg', -1)
    beds.append(bed)
    surf = get_nc_data('icecap_output_vel_{}.nc'.format(i), 'usurf', -1)
    surfs.append(surf)
    surfs_in.append(np.maximum(true_surf + np.random.normal(0, x, np.shape(true_surf)), true_bed))
    vols.append(np.sum(surf-bed))
    taucs.append(np.nanmedian(get_nc_data('icecap_output_vel_{}.nc'.format(i), 'tauc', 0)[mask==1]))
    vels.append(np.nanmean(get_nc_data('icecap_output_vel_{}.nc'.format(i), 'velbar_mag', 0)[mask==1]))
    fluxes.append(np.sum(get_nc_data('icecap_output_vel_{}.nc'.format(i), 'flux_mag', 0)[mask==1]))
    bed_deviations.append(rmse(bed[mask==1], reference_bed[mask==1]))
    rsmes.append(rmse(bed[mask==1], true_bed[mask==1]))


taucs.append(np.median(reference_tauc[mask==1]))
vels.append(np.nanmean(reference_vel[mask==1]))
fluxes.append(np.sum(reference_flux))
xs.append(1)
vols.append(reference_vol)
bed_deviations.append(rmse(reference_bed[mask==1], true_bed[mask==1]))

fig, ax = plt.subplots()
line = ax.plot(xs, [no_sliding_vol/true_vol]*len(xs), '--', c='black')
plt.text(0.8, no_sliding_vol/true_vol+0.01, 'ice volume if no sliding anywhere')
points = ax.scatter(xs, vols/true_vol, label = 'applying friction updates', c=bed_deviations, cmap='plasma', vmin = 0, vmax = np.max(bed_deviations))
ax.set_xlabel('velocity / true velocity')
ax.set_ylabel('modelled ice volume / true ice volume')
cbar = fig.colorbar(points, ax = ax)
cbar.ax.set_ylabel('RMSE')
ax1 = plt.axes([0,0,1,1])
ip1 = InsetPosition(ax, [0.1,0.5,0.3,0.3])
ax1.set_axes_locator(ip1)
ax1.pcolor(beds[0], vmin = -500, vmax = 1000)
ax1.axis('off')
ax2 = plt.axes([0,0,.99,1])
ip2 = InsetPosition(ax, [0.65,0.4,0.3,0.3])
ax2.set_axes_locator(ip2)
ax2.pcolor(beds[-1], vmin = -500, vmax = 1000)
ax2.axis('off')
plt.savefig('./figures/velocity_errors_v1.1.png')
plt.show()

model = LinearRegression().fit(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations)))

dts = [1.0, .5, .25, .15, .05, .01]
betas = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0]
thetas = [0.0, 0.015, 0.05,0.1, 0.5]
bed_misfits = []
dh_misfits = []
beds = []
beds_S = []
for theta in thetas:
    bed_misfits.append(np.loadtxt('icecap_output_theta_{}_B.csv'.format(theta)))
    dh_misfits.append(np.loadtxt('icecap_output_theta_{}_dh.csv'.format(theta)))
    beds.append(get_nc_data('icecap_output_theta_{}.nc'.format(theta), 'topg', -1))
    beds_S.append(get_nc_data('icecap_output_theta_{}_S_pert_5.nc'.format(theta), 'topg', -1))

plot_beds = []
[plot_beds.append(i) for i in beds]
plot_beds[1] = reference_bed
betas[1] = 0.025
[plot_beds.append(i) for i in beds_S]
plot_beds[6] = get_nc_data('icecap_output_S_ref_rand_5.0_correct_diffusivity.nc', 'topg', 0)
fig, ax = plot_list(plot_beds, 2, 5, vmin = -500, vmax = 500)
fig.set_size_inches(10,5)
for i,a in enumerate(ax[0,:5]):
    a.set_title('theta = {}'.format(thetas[i]))

plt.savefig('./figures/icecap_theta_sensitivity_v0.1.png')
plt.show()

colormap = plt.cm.copper
colors = [colormap(i) for i in np.linspace(0, 1,len(betas))]
bounds = np.linspace(0, np.max(betas))
norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm))
#colormap1 = plt.cm.Reds
#colors1 = [colormap1(i) for i in np.linspace(0, 1,len(betas))]
fig, ax = plt.subplots(1,2, figsize=(10,4))
#ax1 = ax.twinx()
ax1 = plt.axes([0,0,1,1])
ip1 = InsetPosition(ax[0], [0.5,0.5,0.4,0.35])
ax1.set_axes_locator(ip1)
ax2 = plt.axes([0,0,.99,1])
ip2 = InsetPosition(ax[1], [0.45,0.5,0.4,0.35])
ax2.set_axes_locator(ip2)
for i in range(len(bed_misfits)):
    line = ax[0].plot(bed_misfits[i], color = colors[i], label = 'K = {}'.format(betas[i]))
    line_in = ax1.plot(range(8000,10000), bed_misfits[i][8000:10000], color = colors[i])
    line1 = ax[1].plot(dh_misfits[i], '--', color = colors[i], label = 'K = {}'.format(betas[i]))
    line_in = ax2.plot(range(8000,10000), dh_misfits[i][8000:10000], '--', color = colors[i])

ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[0].set_xlabel('iteration')
ax[1].set_xlabel('iteration')
ax[0].set_ylabel('bed misfit (m)')
ax[1].set_ylabel('dh/dt misfit (m)')
ax[0].legend(loc=2, bbox_to_anchor=(0.9,0.9))
#ax[1].legend()
#cbar = fig.colorbar(sm, ax = ax[0])
#cbar1 = fig.colorbar(sm, ax = ax[1])
#plt.savefig('icecap_K_sensitivity_v1.0.png')
plt.show()

END
