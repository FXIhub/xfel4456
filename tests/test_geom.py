import matplotlib.pyplot as plt
import extra_geom
import h5py
import numpy as np

PREFIX      = '/gpfs/exfel/exp/SPB/202302/p004456/'

powder_fnam = PREFIX + 'scratch/powder/powder_r0040.h5'
geom_fnam   = PREFIX + 'usr/geometry/geom_v5.geom'

# load powder
with h5py.File(powder_fnam) as f:
    powder = f['data'][()]

# geom file 
geom = extra_geom.JUNGFRAUGeometry.from_crystfel_geom(geom_fnam)

# to get assembled image (2D)
image, centre = geom.position_modules(powder)

# plot with pyqtgraph

#import pyqtgraph as pg
#pg.show(image)

# plot powder directly from data using extra_geom with matplotlib
fig, ax = plt.subplots()
ax = geom.plot_data(powder, axis_units='m', ax = ax, vmin=0, vmax=400000)
#plt.savefig('powder_r0040.svg')
plt.show()
