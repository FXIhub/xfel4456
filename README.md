# Analysis code for p004456

Experiment directory:
```
/gpfs/exfel/exp/SPB/202302/p004456
```

Put personal analysis code in usr/Shared, e.g. for user "amorgan"
```
/gpfs/exfel/exp/SPB/202302/p004456/usr/Shared/amorgan
```

Put any analysis files (bigger than a few megabytes) in /scratch:
```
/gpfs/exfel/exp/SPB/202302/p004456/scratch
```


## analysis goals:
- [x] VDS: virtual h5 files that put all processed module data into a single cxi file.
    - e.g. ```sbatch --array=5 offline/slurm/vds_array.sh``` to generate a vds file for run 5 from processed data.
- [x] Generate powder pattern
    - e.g. ```sbatch --array=40 offline/slurm/powder_sum_array.sh``` to sum all frames in run 40 using vds file.
- [x] Get initial geometry (and test with extra-geom)
    - There is a ```usr/geometry/geom_v5.geom``` crystFEL geom file (probably from Oleksandr).
    - To have a look ```python tests/test_geom.py``` (output below)
- [x] Jungfrau mask maker
    - Example ```python offline/tools/maskmaker.py /gpfs/exfel/exp/SPB/202302/p004456/scratch/powder/powder_r0040.h5/data```
- [x] Calculate number, location and intensity of streaks
    - Example ```sbatch --array=40 offline/slurm/streak_finder.sh```.
      This outputs an "events" file here:
      ```
      h5ls -r /p004456/scratch/events/events_r0040.h5
      /                        Group
      /litpixels               Dataset {6066}
      /streak_list             Dataset {80173/Inf}
      /streaks                 Dataset {6066}
      /total_intens            Dataset {6066}
      ```
- [x] Generate streakogram, and virtual powder from above
    - Example ```python offline/streaks_powder_plotter.py -e  "/gpfs/exfel/u/scratch/SPB/202302/p004456/events/events_r0040.h5" -g "/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom"```
    will create `events_r0040.streaks_powder_plot.png` and `events_r0040.streakogram_plot.png` in folder `/gpfs/exfel/u/scratch/SPB/202302/p004456/events/`
- [ ] Save hits to cxi file
- [x] Write cxi file from scan runs, for the purpose of speckle-tracking
    - It seems Margarita already has a means of streaming scan data, with a modified scan log, to the ST software. It would be nice to include this code here.



### powder run 40 (test data)
![powder_r0040.h5](/tests/powder_r0040.svg)

### mask maker 40 (test data)
![mask_maker](/tests/mask_maker_test.png)
