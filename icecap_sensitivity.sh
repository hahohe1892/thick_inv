#!/bin/bash

for S_ref_rand in 0.02 0.04
do
    mpiexec -n 1 python3 icecap_inversion.py -S_ref_rand $S_ref_rand
done


: <<'END'

import numpy as np
from funcs import *
from bed_inversion import *
from sklearn.linear_model import LinearRegression

bed_deviations = []
bed_deviations_tf = []
taucs = []
vels = []
vols = []
vols_tf = []
fluxes = []
true_bed = get_nc_data('ice_build_output.nc', 'topg', -1)
true_surf = get_nc_data('ice_build_output.nc', 'usurf', -1)
true_vol = np.sum(true_surf - true_bed)
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
    surf = get_nc_data('icecap_output_mb_{}.nc'.format(i), 'usurf', -1)
    surf_tf =  get_nc_data('icecap_output_mb_{}_true_friction.nc'.format(i), 'usurf', -1)
    vols.append(np.sum(surf-bed))
    vols_tf.append(np.sum(surf_tf-bed_tf))
    taucs.append(np.nanmean(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'tauc', 0)[mask==1]))
    vels.append(np.nanmean(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'velbar_mag', 0)[mask==1]))
    fluxes.append(np.sum(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'flux_mag', 0)[mask==1]))
    bed_deviations.append(np.nanmean((bed[mask==1] - reference_bed[mask==1])))
    bed_deviations_tf.append(np.nanmean((bed_tf[mask==1] - reference_bed[mask==1])))

xs.append(1)
fig, ax = plt.subplots()
taucs.append(np.mean(reference_tauc[mask==1]))
vels.append(np.nanmean(reference_vel[mask==1]))
vols.append(reference_vol)
vols_tf.append(reference_vol)
bed_deviations_tf.append(0)
fluxes.append(np.sum(reference_flux))
points1 = ax.scatter(xs,vols_tf/reference_vol, c = 'orange', label='true friction field prescribed')
line1 = ax.plot(np.sort(xs), np.sort(np.array(xs)**(1/5)), '--', c = 'orange', label='5th root of mass balance errors')
#ax1 = ax.twinx()
bed_deviations.append(0)
points = ax.scatter(xs, vols/reference_vol, label='friction updates are applied')
line = ax.plot(np.sort(xs), np.sort(xs), '--', label='1:1 line')
ax.legend()
ax.set_xlabel('mass balance / true mass balance')
ax.set_ylabel('modelled ice volume / true ice volume')
plt.savefig('./figures/mass_balance_errors.png')
plt.show()



model = LinearRegression().fit(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations)))

END
