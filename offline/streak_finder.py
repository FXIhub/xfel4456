import sys
import multiprocessing as mp

import numpy as np
import h5py
from tqdm import tqdm

from hit_finding_utils import single_streak_finder
from constants import PREFIX, VDS_DATASET, VDS_MASK_DATASET
from constants import FRAME_SHAPE, ADU_PER_PHOTON
from constants import STREAK_PTHRESHOLD, STREAK_MIN_PIX

class StreakHitFinder():
    def __init__(self, run):
        self.vds_fname = PREFIX + 'vds/r%.4d.cxi' % run
        self.output_fname = PREFIX + 'events/r%.4d_events.h5' % run

        with h5py.File(self.vds_fname, 'r') as f:
            self.num_events = f['entry_1/trainId'].shape[0]

        # streak list:
        # frame index, pixel_coord, q,        intensity, area
        # uint32     , uint32     , float32,  float32,   uint16
        self.streak_dtype = np.dtype([('frame_index', np.uint32),
                                      ('pixel_coord', np.uint32),
                                      ('intensity', np.float32),
                                      ('area', np.uint16)])

    def run(self):
        #nproc = mp.cpu_count()
        nproc = 8
        indices = np.arange(self.num_events)
        # split frames over ranks
        ev_rank = np.linspace(0, self.num_events, nproc+1).astype(int)

        lock = mp.Lock()

        with h5py.File(self.output_fname, 'a') as fptr:
            self.copy_ids(self.vds_fname, fptr)
            self._init_datasets(fptr['entry_1'])

        jobs = []
        for m in range(nproc):
            job = mp.Process(target=self._worker,
                             args=(m, indices[ev_rank[m]:ev_rank[m+1]], lock))
            jobs.append(job)
            job.start()
        [j.join() for j in jobs]
        print('DONE')
        sys.stdout.flush()

    def _init_datasets(self, h5group):
        if 'litpixels' in h5group:
            del h5group['litpixels']
        if 'total_intens' in h5group:
            del h5group['total_intens']
        if 'streaks' in h5group:
            del h5group['streaks']
        if 'streak_list' in h5group:
            del h5group['streak_list']
        h5group.create_dataset('litpixels', shape=(self.num_events,), dtype=np.uint32)
        h5group.create_dataset('total_intens', shape=(self.num_events,), dtype=np.float32)
        h5group.create_dataset('streaks', shape=(self.num_events,), dtype=np.uint16)
        h5group.create_dataset('streak_list', shape=(0,), maxshape=(None,), dtype=self.streak_dtype)

    def _worker(self, rank, my_indices, lock):
        N = len(my_indices)

        print('Rank %d: Processing indices %d to %d' % (rank, my_indices.min(), my_indices.max()))
        sys.stdout.flush()

        if rank == 0 :
            it = tqdm(range(N), desc = f'Processing data from {self.vds_fname}')
        else :
            it = range(N)

        frame = np.empty(FRAME_SHAPE, dtype='f4')
        bad_pix = np.empty(FRAME_SHAPE, dtype='bool')

        litpixels    = np.zeros((N,), dtype = int)
        total_intens = np.zeros((N,), dtype = float)
        streaks      = np.zeros((N,), dtype = int)

        streak_list    = np.zeros((N,), dtype = self.streak_dtype)
        streak_index   = 0

        with h5py.File(self.vds_fname) as fptr:
            data_dset = fptr[VDS_DATASET]
            mask_dset = fptr[VDS_MASK_DATASET]

            for i in it:
                index = my_indices[i]
                frame[:] = data_dset[index]
                bad_pix[:] = (mask_dset[index] != 0)

                # corrected jungfrau has weird values
                bad_pix[~np.isfinite(frame)] = True
                frame[bad_pix] = 0

                # photon conversion
                frame = np.clip(np.round(frame/ADU_PER_PHOTON-0.3).astype('i4'), 0, None)

                # determine threshold
                percentile_threshold = np.percentile(frame, STREAK_PTHRESHOLD)

                # find streaks in each module
                for m in range(FRAME_SHAPE[0]):
                    (
                        label_filtered_sorted,
                        weighted_centroid_filtered,
                        props,
                        _,
                        all_labels,
                    ) = single_streak_finder(img_array=frame[m],
                                             thld=percentile_threshold,
                                             min_pix=STREAK_MIN_PIX)
                    
                    # number of streaks
                    streaks[i] += len(label_filtered_sorted)

                    for l, (li, lj) in zip(label_filtered_sorted, weighted_centroid_filtered) :
                        pc = m * FRAME_SHAPE[1] * FRAME_SHAPE[2] + FRAME_SHAPE[2] * int(round(li)) + int(round(lj))
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
                total_intens[i] = np.sum(frame)

        # take turns writing frame_buf to file

        # write to file sequentially
        if rank == 0:
            print('Writing photons to     ', self.output_fname)
            sys.stdout.flush()

        if lock.acquire() :
            with h5py.File(self.output_fname, 'a') as f:
                st = my_indices.min()
                en = my_indices.max() + 1
                g = f['entry_1']
                g['litpixels'][st:en]    = litpixels
                g['total_intens'][st:en] = total_intens
                g['streaks'][st:en]      = streaks

                # resize then write streak list
                s = g['streak_list'].shape[0]
                g['streak_list'].resize(s + streak_index, axis=0)
                g['streak_list'][s: s + streak_index] = streak_list[:streak_index]

            print(f'rank {rank} done')
            sys.stdout.flush()
            lock.release()

    @staticmethod
    def copy_ids(vds_fname, fptr):
        print('Copying IDs from VDS file')
        sys.stdout.flush()

        if 'entry_1/trainId' in fptr: del fptr['entry_1/trainId']
        if 'entry_1/cellId' in fptr: del fptr['entry_1/cellId']

        with h5py.File(vds_fname, 'r') as f_vds:
            fptr['entry_1/trainId'] = f_vds['entry_1/trainId'][:]
            fptr['entry_1/cellId'] = f_vds['entry_1/cellId'][:]

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Find streaks and output list to scratch/events/r<run>_events.h5. Requires processed data and vds file.')
    parser.add_argument('run', type=int, help='Run number')
    args = parser.parse_args()

    hitfinder = StreakHitFinder(args.run)
    hitfinder.run()

if __name__ =='__main__':
    main()
