#!/bin/bash

for mb_factor in .25 .5 .75 1.25 1.5 1.75
do
    mpiexec -n 1 python3 icecap_inversion.py -mb_factor $mb_factor&
done


: <<'END'
import numpy as np
from funcs import *
from bed_inversion import *
from sklearn.linear_model import LinearRegression

bed_deviations = []
reference_bed = get_nc_data('icecap_output_mb_1.0.nc', 'topg', -1).data
mask = get_nc_data('ice_build_output.nc', 'mask', -1).data/2
mb_factors = np.array([0.25, .5, .75, 1.0, 1.25, 1.5])
for i in mb_factors:
	 bed = get_nc_data('icecap_output_mb_{}.nc'.format(i), 'topg', -1).data
	 bed_deviations.append(np.nanmean((bed[mask==1] - reference_bed[mask==1])))

plt.scatter(mb_factors, bed_deviations)
plt.show()

model = LinearRegression().fit(mb_factors.reshape((-1,1)), np.array(bed_deviations))
print(model.score(mb_factors.reshape((-1,1)), np.array(bed_deviations)))

END
