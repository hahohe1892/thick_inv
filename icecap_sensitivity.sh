#!/bin/bash

for vel_factor in .25 .5 .75 1.25 1.5 1.75
do
    mpiexec -n 1 python3 icecap_inversion.py -vel_factor $vel_factor
done


: <<'END'
import numpy as np
from funcs import *
from bed_inversion import *
from sklearn.linear_model import LinearRegression

bed_deviations = []
reference_bed = get_nc_data('icecap_output_mb_1.0.nc', 'topg', -1).data
mask = get_nc_data('ice_build_output.nc', 'mask', -1).data/2
vel_factors = [0.25, .5, .75, 1.25, 1.5, 1.75]
for i in vel_factors:
	 bed = get_nc_data('icecap_output_vel_{}.nc'.format(i), 'topg', -1).data
	 bed_deviations.append(np.nanmean((bed[mask==1] - reference_bed[mask==1])))

vel_factors.append(1)
bed_deviations.append(1)
plt.scatter(vel_factors, bed_deviations)
plt.show()

model = LinearRegression().fit(np.array(vel_factors).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(vel_factors).reshape((-1,1)), np.array(bed_deviations)))

END
