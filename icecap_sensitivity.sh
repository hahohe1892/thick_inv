#!/bin/bash

for ice_density in 775 800 825 850 875 925
do
    mpiexec -n 1 python3 icecap_inversion.py -ice_density $ice_density
done


: <<'END'

import numpy as np
from funcs import *
from bed_inversion import *
from sklearn.linear_model import LinearRegression

bed_deviations = []
reference_bed = get_nc_data('icecap_output_mb_1.0.nc', 'topg', -1).data
mask = get_nc_data('ice_build_output.nc', 'mask', -1).data/2
ice_densitys = [775, 800, 825, 850, 875, 925]
for i in ice_densitys:
    bed = get_nc_data('icecap_output_ice_density_{}.nc'.format(i), 'topg', -1).data
    bed_deviations.append(np.nanmean((bed[mask==1] - reference_bed[mask==1])))

ice_densitys.append(900)
bed_deviations.append(0)
plt.scatter(ice_densitys, bed_deviations)
plt.show()

model = LinearRegression().fit(np.array(ice_densitys).reshape((-1,1)), np.array(bed_deviations))
print(model.score(np.array(ice_densitys).reshape((-1,1)), np.array(bed_deviations)))

END
