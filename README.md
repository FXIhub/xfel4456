# Ananysis code for p004456

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
- [ ] Calculate number, location and intensity of streaks
- [ ] Generate streakogram, and virtual powder from above
- [ ] Save hits to cxi file
- [ ] Write cxi file from scan runs, for the purpose of speckle-tracking



### powder run 40 (test data)
![powder_r0040.h5](/tests/powder_r0040.svg)
