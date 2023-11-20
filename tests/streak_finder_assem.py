'''
streak finding for the assembled image from JUNGFRAU detector.


'''

import argparse

# DATA_PATH   = 'entry_1/instrument_1/detector_1/data'
# MASK_PATH   = 'entry_1/instrument_1/detector_1/mask'
#
# parser = argparse.ArgumentParser(description='Find streaks and output list to scratch/events/events_r<run>.h5. Requires processed data and vds file.')
# parser.add_argument('run', type=int, help='Run number')
# parser.add_argument('-a', '--ADU_per_photon',
#                     help='ADUs per photon, for proc/ data the adus are in units of keV',
#                     type=float)
# parser.add_argument('--percentile_threshold',
#                     help='determines the threshold value for streak finding',
#                     type=float, default = 99.5 )
# parser.add_argument('--min_pix',
#                     help='determines the minimum number of connected pixels for a streak',
#                     type=int, default = 15 )
#
# args = parser.parse_args()
#
# args.vds_file         = PREFIX+'scratch/vds/r%.4d.cxi' %args.run
# args.output_file      = PREFIX+'scratch/events/events_r%.4d.h5'%args.run


import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__))+'/offline')
import numpy as np
import h5py
import CBD_detector_Jungfrau_utils0 as CBD_ut
from tqdm import tqdm
import time

from hit_finding_utils import single_streak_finder
import multiprocessing as mp
import pickle as pk
from skimage import measure, morphology, feature

PREFIX      = '/gpfs/exfel/exp/SPB/202302/p004456/'
proposal = 700000
run_id = 39
geom_file = '/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom'
train_img_dict = CBD_ut.read_train(proposal,run_id,0,\
geom_file=geom_file,geom_assem='True',ROI=(0,2400,0,2400))
no_trains = train_img_dict['no_trains']
IMG_SHAPE = train_img_dict['adc_img'].shape[1:]

n_procs = mp.cpu_count()
chunk_size = int(np.ceil(no_trains/n_procs))
thld = 10
min_pix = 5
max_pix = 10000
min_peak = 5
mask_file='None'
Region='ALL'

print('setup done!')
def worker(rank,no_trains):
    # intialise output flie
    output_log_file = PREFIX + f'/scratch/usr/Shared/cli/run_{run_id:d}_log_worker{rank:d}.txt'
    output_pickle_file = PREFIX + f'/scratch/usr/Shared/cli/run_{run_id:d}_peak_info_worker{rank:d}.pkl'
    output_evt_lst_file = PREFIX + f'/scratch/usr/Shared/cli/run_{run_id:d}_evt_lst_worker{rank:d}.txt'
    if os.path.exists(output_log_file):
        print('Deleting existing log file:', output_log_file)
        os.remove(output_log_file)

    if os.path.exists(output_pickle_file):
        print('Deleting existing pkl file:', output_pickle_file)
        os.remove(output_pickle_file)

    if os.path.exists(output_evt_lst_file):
        print('Deleting existing event list file:', output_evt_lst_file)
        os.remove(output_evt_lst_file)

    worker_frame_id_lst = np.arange(chunk_size*rank,chunk_size*(rank+1))
    worker_frame_id_lst = worker_frame_id_lst.tolist()
    _ = \
    List_streak_finder(rank,worker_frame_id_lst,output_log_file,output_pickle_file,output_evt_lst_file,\
    thld,min_pix,max_pix,min_peak,mask_file=mask_file,Region=Region)


def List_streak_finder(rank,frame_id_lst,output_log_file,output_pickle_file,output_evt_lst_file,thld,min_pix,max_pix,min_peak,mask_file='None',Region='ALL'):

    if len(frame_id_lst)==0:
        sys.exit(f'No frames found')


    if Region=='ALL':
        x_min=0
        y_min=0
        x_max=IMG_SHAPE[0]
        y_max=IMG_SHAPE[1]
    elif Region=='Q':
        x_min=np.round(0.5*IMG_SHAPE[0]).astype(int)
        y_min=np.round(0.5*IMG_SHAPE[1]).astype(int)
        x_max=IMG_SHAPE[0]
        y_max=IMG_SHAPE[1]
    elif Region=='C':
        x_min=np.round(0.25*IMG_SHAPE[0]).astype(int)
        y_min=np.round(0.25*IMG_SHAPE[1]).astype(int)
        x_max=np.round(0.75*IMG_SHAPE[0]).astype(int)
        y_max=np.round(0.75*IMG_SHAPE[1]).astype(int)
    else:
        sys.exit('Check the Region option: ALL,Q,C')

    if mask_file!='None':
        mask_file=os.path.abspath(mask_file)
        m=h5py.File(mask_file,'r')
        mask=m['/data/data'][x_min:x_max,y_min:y_max].astype(bool)
        m.close()
    elif mask_file=='None':
        mask=np.ones((IMG_SHAPE[0],IMG_SHAPE[1])).astype(bool)
        mask=mask[x_min:x_max,y_min:y_max]
    else:
        sys.exit('the mask file option is inproper.')
    HIT_counter=0
    HIT_frame_id_list=[]
    peakXPosRaw=np.zeros((len(frame_id_lst),1024))
    peakYPosRaw=np.zeros((len(frame_id_lst),1024))
    pixel_size=float(75e-6)#to be changed
    nPeaks=np.zeros((len(frame_id_lst),),dtype=np.int16)
    peakTotalIntensity=np.zeros((len(frame_id_lst),1024))

    lf=open(output_log_file,'w',1)
    ef=open(output_evt_lst_file,'w',1)

    lf.write(f'run : {run_id:d}, worker_rank: {rank:}')
    lf.write('\n-----------')
    lf.write('\nthld: %d\nmin_pix: %d\nmax_pix: %d\nmin_peak: %d\nmask_file: %s\nRegion: %s'%\
    (thld,min_pix,max_pix,min_peak,mask_file,Region))
    lf.write('\n-----------')

    for event_no in range(len(frame_id_lst)):
        frame_id = frame_id_lst[event_no]
        bkg = 0

        train_img_dict = CBD_ut.read_train(proposal,run_id,frame_id,\
        geom_file=geom_file,geom_assem='True',ROI=(0,2400,0,2400))

        img_array = train_img_dict['adc_img'][0]

        img_array_bgd_subtracted = img_array - bkg
        bimg_masked = (img_array_bgd_subtracted > thld) * mask

        all_labels = measure.label(bimg_masked, connectivity=1)
        # all_labels=measure.label(bimg_masked,connectivity=1) #connectivity is important here, for sim data,use 2, for exp data use 1
        props = measure.regionprops(all_labels, img_array_bgd_subtracted)

        area = np.array([r.area for r in props]).reshape(
            -1,
        )

        label = np.array([r.label for r in props]).reshape(
            -1,
        )
        #     label_filtered=label[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
        label_filtered = label[(area > min_pix) * (area < max_pix)]
        #     area_filtered=area[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
        area_filtered = area[(area > min_pix) * (area < max_pix)]
        area_sort_ind = np.argsort(area_filtered)[::-1]
        label_filtered_sorted = label_filtered[area_sort_ind]

        # weighted_centroid_filtered = np.zeros((len(label_filtered_sorted), 2))
        # for index, value in enumerate(label_filtered_sorted):
        #     weighted_centroid_filtered[index, :] = np.array(
        #         props[value - 1].weighted_centroid
        #     )
        weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
        max_intensity_filtered=np.zeros((len(label_filtered_sorted),1))
        mean_intensity_filtered=np.zeros((len(label_filtered_sorted),1))
        area_filtered = np.zeros((len(label_filtered_sorted),1))
        peak_no=0
        for index,value in enumerate(label_filtered_sorted):
            weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
            max_intensity_filtered[index,:]=props[value-1].max_intensity
            mean_intensity_filtered[index,:]=props[value-1].mean_intensity
            area_filtered[index,:]=props[value-1].area
            peak_no=len(label_filtered_sorted)
            nPeaks[event_no]=np.int16(np.minimum(1024,peak_no))
            # print(nPeaks[event_no])
            # print(peakTotalIntensity.shape)
            # print(mean_intensity_filtered.shape)
            peakTotalIntensity[event_no,:nPeaks[event_no]]=(mean_intensity_filtered[:nPeaks[event_no],0].reshape(-1,))\
            *(area_filtered[:nPeaks[event_no],0].reshape(-1,))
            peakXPosRaw[event_no,:nPeaks[event_no]]=weighted_centroid_filtered[:nPeaks[event_no],0]+x_min
            peakYPosRaw[event_no,:nPeaks[event_no]]=weighted_centroid_filtered[:nPeaks[event_no],1]+y_min


        if (peak_no>=min_peak) and (peak_no<=1024):
            HIT_counter+=1
            HIT_frame_id_list.append(frame_id)


            print('HIT!!!!  frame %d: %d peaks found'%(frame_id,peak_no))
            lf.write('\nHIT!!!!  Frame %d: %d peaks found\n'%(frame_id,peak_no))
            ef.write(f'frame {frame_id:d} \n')
        else:
            print('BLANK!   Frame %d: %d peaks found'%(frame_id,peak_no))
            lf.write('\nBLANK!   Frame %d: %d peaks found'%(frame_id,peak_no))
            #pass


    peak_list_dict={'geom_file':geom_file,'frame_id_lst':frame_id_lst,'nPeaks':nPeaks.astype(np.int16),'peakTotalIntensity':peakTotalIntensity,\
            'peakXPosRaw':peakXPosRaw,'peakYPosRaw':peakYPosRaw}
    pf=open(output_pickle_file,'wb')
    pk.dump(peak_list_dict,pf)
    pf.close()
    print(f'run: {run_id:d}\n')
    print(f'frame_id: \n')
    print(frame_id_lst)
    print('%d   out of  %d  hits found!'%(HIT_counter,len(frame_id_lst)))
    print('HIT rate: %.2f %%'%(100*HIT_counter/len(frame_id_lst)))
    print('HIT events:')
    print(HIT_frame_id_list)
    print(f'Peak information saved in {output_pickle_file:s}')

    total_event_no=len(frame_id_lst)
    hit_rate=100*HIT_counter/total_event_no


    lf.write('\n------------------------------------------------------------------')
    lf.write(f'\nframe: {frame_id:d}')
    lf.write('\n %d   out of  %d  hits found!'%(HIT_counter,total_event_no))
    lf.write('\n HIT rate: %.2f %%'%(hit_rate))
    #lf.write('\n HIT events:\n')
    lf.write('\n------------------------------------------------------------------')
    #for event in HIT_event_no_list:
    #lf.write('%d \n'%event)
    #lf.write('-----------------')
    lf.close()
    ef.close()

    return total_event_no, HIT_counter, hit_rate, HIT_frame_id_list

print(f'n_procs: {n_procs:d}')
jobs = [mp.Process(target=worker, args=(m, no_trains)) for m in range(n_procs)]
[j.start() for j in jobs]
[j.join() for j in jobs]
print('Done')
sys.stdout.flush()