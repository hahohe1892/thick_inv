#!/bin/bash

for ice_temp in 264 266 270 272
do
    mpiexec -n 1 python3 icecap_inversion.py -ice_temp $ice_temp
done


: <<'END'

def calc_ice_temps(T):
    return 1.733e3*np.exp(-13.9e4/(8.3*T)

import numpy as np
from funcs import *
from bed_inversion import *
from sklearn.linear_model import LinearRegression

bed_deviations = []
reference_bed = get_nc_data('icecap_output_mb_1.0.nc', 'topg', -1).data
mask = get_nc_data('ice_build_output.nc', 'mask', -1).data/2
ice_temps = [264.0, 266.0, 268.0, 270.0, 272.0]
for i in ice_temps:
    bed = get_nc_data('icecap_output_ice_temp_{}.nc'.format(i), 'topg', -1).data
    bed_deviations.append(np.nanmean((bed[mask==1] - reference_bed[mask==1])))

#ice_temps.append(268)
#bed_deviations.append(1)
plt.scatter(ice_temps, bed_deviations)
plt.show()

model = LinearRegression().fit(np.array(ice_temps).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(ice_temps).reshape((-1,1)), np.array(bed_deviations)))

END
