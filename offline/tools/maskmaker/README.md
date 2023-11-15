# Graphical Mask Maker for Jungfrau detector 
Based on CsPadMaskMaker.

The boolean mask is saved to:
    mask.h5
        entry_1/good_pixels
        entry_1/bad_pixels


## Example
exfel-python has all of the dependencies built in:
```
$ module load exfel exfel-python
$ python maskmaker.py /gpfs/exfel/exp/SPB/202302/p004456/scratch/powder/powder_r0040.h5/data
```
