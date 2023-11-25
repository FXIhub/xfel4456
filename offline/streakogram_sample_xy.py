import argparse

parser = argparse.ArgumentParser(description='Plot a 2d histogram showing the total peak intensity vs sample x and y position from an events.h5 file. \nOutputs to scratch/events/<run_no>_streak_intensity_heatmap_xy.png')
parser.add_argument('run', help='Run number', type=int)
args = parser.parse_args()


import h5py
import numpy as np
import matplotlib.pyplot as plt

run = args.run

from constants import PREFIX

out_fnam = PREFIX + 'events/r%.4d_streak_intensity_heatmap_xy.png'%run

# get events file
events_fnam = PREFIX + 'events/r%.4d_events.h5'%run

with h5py.File(events_fnam) as f:
    streak_list = f['entry_1/streak_list'][()]
    trainID     = f['entry_1/trainId'][()]
    x = f['entry_1/sample_pos_mm_x'][()]
    y = f['entry_1/sample_pos_mm_y'][()]

# get total streak intensities per frame
frame_streak_intensity = np.bincount(streak_list['frame_index'], weights = streak_list['intensity'], minlength=trainID.shape[0])

# now generate 2D histogram
bins = int(round(x.shape[0]**0.5/3))
H, yedges, xedges = np.histogram2d(y, x, bins=bins, density=True, weights=frame_streak_intensity)

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

vmin = 0
vmax = np.percentile(H, 95)
im = ax.imshow(H, origin='lower', extent= (xedges[0], xedges[-1], yedges[-1], yedges[0]), cmap='Greys_r', vmin=vmin, vmax=vmax, aspect='auto')
ax.set_xlabel('x sample position (mm)')
ax.set_ylabel('y sample position (mm)')
ax.set_title(f'total streak intensity vs sample position run {run}')

ax.scatter(x, y, s = 1)

plt.colorbar(im)
print('saving plot to', out_fnam)
plt.savefig(out_fnam)
plt.show()
