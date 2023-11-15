import argparse

PREFIX      = '/gpfs/exfel/exp/SPB/202302/p004456/'
DATA_PATH   = 'entry_1/instrument_1/detector_1/data'
MASK_PATH   = 'entry_1/instrument_1/detector_1/mask'

parser = argparse.ArgumentParser(description='Find streaks and output list to scratch/events/events_r<run>.h5. Requires processed data and vds file.')
parser.add_argument('run', type=int, help='Run number')
parser.add_argument('-a', '--ADU_per_photon',
                    help='ADUs per photon, for proc/ data the adus are in units of keV',
                    type=float)
parser.add_argument('--percentile_threshold',
                    help='determines the threshold value for streak finding',
                    type=float, default = 99.5 )
parser.add_argument('--min_pix',
                    help='determines the minimum number of connected pixels for a streak',
                    type=int, default = 15 )

args = parser.parse_args()

args.vds_file         = PREFIX+'scratch/vds/r%.4d.cxi' %args.run
args.output_file      = PREFIX+'scratch/events/events_r%.4d.h5'%args.run


import numpy as np
import h5py
#import extra_data
from tqdm import tqdm
import sys
import time
import os

from hit_finding_utils import single_streak_finder

import multiprocessing as mp


with h5py.File(args.vds_file) as f:
    cellID    = f['/entry_1/cellId'][:, 0]

Nevents = cellID.shape[0]

# intialise output flie
if os.path.exists(args.output_file):
    print('Deleting existing events file:', args.output_file)
    os.remove(args.output_file)

# streak list:
# frame index, pixel_coord, q,        intensity, area
# uint32     , uint32     , float32,  float32,   uint16
streak_dtype = np.dtype([('frame_index', np.uint32), 
                         ('pixel_coord', np.uint32), 
                         ('intensity', np.float32), 
                         ('area', np.uint16)])

with h5py.File(args.output_file, 'w') as f:
    f.create_dataset('litpixels', shape = (Nevents,), dtype = np.uint32)
    f.create_dataset('total_intens', shape = (Nevents,), dtype = np.float32)
    f.create_dataset('streaks', shape = (Nevents,), dtype = np.uint16)
    f.create_dataset('streak_list', shape = (0,), maxshape = (None,), dtype = streak_dtype)

indices = np.arange(Nevents)

size = mp.cpu_count()

# split frames over ranks
events_rank = np.linspace(0, Nevents, size+1).astype(int)

frame_shape = (8,512,1024)

def worker(rank, lock):
    my_indices = indices[events_rank[rank]: events_rank[rank+1]] 

    N = len(my_indices) 

    print(f'rank {rank} is processing indices {events_rank[rank]} to {events_rank[rank+1]}')
    sys.stdout.flush()

    if rank == 0 :
        it = tqdm(range(N), desc = f'Processing data from {args.vds_file}')
    else :
        it = range(N)

    frame    = np.empty(frame_shape, dtype = float)
    bad_pix  = np.empty(frame_shape, dtype = bool)
    
    litpixels    = np.zeros((N,), dtype = int)
    total_intens = np.zeros((N,), dtype = float)
    streaks      = np.zeros((N,), dtype = int)
    
    streak_list    = np.zeros((N,), dtype = streak_dtype)
    streak_index   = 0
    
    with h5py.File(args.vds_file) as g:
        data = g[DATA_PATH]
        mask = g[MASK_PATH]
        
        for i in it:
            index = my_indices[i]
            frame[:]   = np.squeeze(data[index]).astype(float)
            bad_pix[:] = np.squeeze(mask[index]) != 0

            # corrected jungfrau has wierd values
            bad_pix[~np.isfinite(frame)] = True
            
            # photon conversion
            if args.ADU_per_photon :
                frame -= args.ADU_per_photon//2 + 1
                frame /= args.ADU_per_photon
            
            frame[frame<0] = 0
            frame[bad_pix] = 0
            
            # determine threshold
            percentile_threshold = np.percentile( frame, args.percentile_threshold )
            
            # find streaks in each module
            for m in range(frame_shape[0]):
                (
                    label_filtered_sorted,
                    weighted_centroid_filtered,
                    props,
                    _,
                    all_labels,
                ) = single_streak_finder(
                    img_array = frame[m], thld = percentile_threshold, min_pix = args.min_pix
                )
                 
                # number of streaks
                streaks[i] += len(label_filtered_sorted)
        
                for l, (li, lj) in zip(label_filtered_sorted, weighted_centroid_filtered) :
                    pc = m * frame_shape[1] * frame_shape[2] + frame_shape[2] * int(round(li)) + int(round(lj))
                    streak_list[streak_index]['frame_index'] = index
                    streak_list[streak_index]['pixel_coord'] = pc
                    streak_list[streak_index]['intensity']   = props[l-1].intensity_mean * props[l-1].area
                    streak_list[streak_index]['area']        = props[l-1].area
                    streak_index += 1

                    if streak_index == streak_list.shape[0] :
                        streak_list = np.resize(streak_list, (streak_index + N,))
            
            # calculate litpixels 
            litpixels[i] = np.sum(frame > 0)
            
            # calculate total intensity 
            total_intens[i] = np.sum(frame > 0)
            
    # take turns writing frame_buf to file 
        
    # write to file sequentially
    if rank == 0: 
        print('Writing photons to     ', args.output_file)
        sys.stdout.flush()
    
    if lock.acquire() :
        with h5py.File(args.output_file, 'a') as f:
            i0 = events_rank[rank]
            i1 = events_rank[rank+1]
            f['litpixels'][i0: i1]    = litpixels
            f['total_intens'][i0: i1] = total_intens
            f['streaks'][i0: i1]      = streaks

            # resize then write streak list
            s = f['streak_list'].shape[0]
            f['streak_list'].resize(s + streak_index, axis=0)
            f['streak_list'][s: s + streak_index] = streak_list[:streak_index]
        
        print(f'rank {rank} done')
        sys.stdout.flush()
        lock.release()


lock = mp.Lock()
jobs = [mp.Process(target=worker, args=(m, lock)) for m in range(size)]
[j.start() for j in jobs]
[j.join() for j in jobs]
print('Done')
sys.stdout.flush()



