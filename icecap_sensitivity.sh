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
taucs = []
vels = []
reference_bed = get_nc_data('icecap_output_mb_1.0.nc', 'topg', -1).data
reference_tauc = get_nc_data('icecap_output_mb_1.0.nc', 'tauc', -1).data
reference_vel  = get_nc_data('icecap_output_mb_1.0.nc', 'velbar_mag', -1).data
mask = get_nc_data('ice_build_output.nc', 'mask', -1).data/2
xs = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]
for i in xs:
    bed = get_nc_data('icecap_output_mb_{}.nc'.format(i), 'topg', -1).data
    taucs.append(np.mean(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'tauc', 0).data[mask==1]))
    vels.append(np.mean(get_nc_data('icecap_output_mb_{}.nc'.format(i), 'velbar_mag', 0).data[mask==1]))
    bed_deviations.append(np.nanmedian((bed[mask==1] - reference_bed[mask==1])))

xs.append(1)
fig, ax = plt.subplots()
taucs.append(np.mean(reference_tauc[mask==1]))
vels.append(np.mean(reference_vel[mask==1]))
points1 = ax.scatter(xs, vels, c = 'orange')
ax.set_ylim([0, 50])
ax1 = ax.twinx()
bed_deviations.append(0)
points = ax1.scatter(xs, bed_deviations)
plt.show()

model = LinearRegression().fit(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(S_ref_rands).reshape((-1,1)), np.array(bed_deviations)))

END
