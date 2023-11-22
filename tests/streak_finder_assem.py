'''
streak finding for the assembled image from JUNGFRAU detector.
'''

import argparse
PREFIX      = '/gpfs/exfel/exp/SPB/202302/p004456/'
# geom_file = '/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom'
geom_file = '/gpfs/exfel/exp/SPB/202302/p004456/usr/geometry/geom_v5.geom'
proposal = 4456
parser = argparse.ArgumentParser(description='Find streaks in assembled image and output list to scratch/peak_info/run<run>.h5.')
parser.add_argument('--run', type=int, help='Run number', required=True)
parser.add_argument('--thld', type= int, help='litpixel threshold', default=10)
parser.add_argument('--min_pix', type=int,help='minimum connected pixels for a streak', default = 5)
parser.add_argument('--max_pix', type=int, help='Maximum pixels', default=10000)
parser.add_argument('--min_peak', type=int, help='Minimum peak',default=5 )
parser.add_argument('--mask_file', type=str, help='Mask file path', default='None')
parser.add_argument('--bkg_file', type=str, help='background file path', default='None')
parser.add_argument('--region', type=str, help='Region: "ALL/Q/C" ', default='ALL')

args = parser.parse_args()

args.output_file      = PREFIX+'scratch/peak_info/run%.4d.h5'%args.run


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
train_img_dict = CBD_ut.read_train(proposal,args.run,0,\
geom_file=geom_file,geom_assem='True',ROI=(0,2400,0,2400))
no_trains = train_img_dict['no_trains']
IMG_SHAPE = train_img_dict['adc_img'].shape[1:]

n_procs = mp.cpu_count()
chunk_size = int(np.ceil(no_trains/n_procs))


print('setup done!')
def worker(rank,no_trains):
    # intialise output flie
    output_log_file = PREFIX + f'/scratch/usr/Shared/butolama/run_{args.run:d}_log_worker{rank:d}.txt'
    output_pickle_file = PREFIX + f'/scratch/usr/Shared/butolama/run_{args.run:d}_peak_info_worker{rank:d}.pkl'
    output_evt_lst_file = PREFIX + f'/scratch/usr/Shared/butolama/run_{args.run:d}_evt_lst_worker{rank:d}.txt'


    for file_path in [output_log_file, output_pickle_file, output_evt_lst_file]:
        if os.path.exists(file_path):
            print(f'Deleting existing file: {file_path}')
            os.remove(file_path)

    worker_frame_id_lst = np.arange(chunk_size*rank,chunk_size*(rank+1)).tolist()
    _ = List_streak_finder(
        rank,
        worker_frame_id_lst,
        output_log_file,
        output_pickle_file,
        output_evt_lst_file,
        thld = args.thld,
        min_pix=args.min_pix,
        max_pix=args.max_pix,
        min_peak=args.min_peak,
        mask_file=args.mask_file,
        bgk_file = args.bkg_file,
        region=args.region)



def List_streak_finder(rank,frame_id_lst,output_log_file,output_pickle_file,output_evt_lst_file,thld,min_pix,max_pix,min_peak,mask_file,bkg_file,region):

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
        mask_file=os.path.abspath(args.mask_file)
        m=h5py.File(mask_file,'r')
        mask=m['/entry_1/goodpixels'][x_min:x_max,y_min:y_max].astype(bool)
        m.close()
        geom = JUNGFRAUGeometry.from_crystfel_geom(geom_file)
        mask, center = geom.position_modules_fast(mask) ### to assmeble the mask from geometry
    elif mask_file=='None':
        mask=np.ones((IMG_SHAPE[0],IMG_SHAPE[1])).astype(bool)
        mask=mask[x_min:x_max,y_min:y_max]
    else:
        sys.exit('the mask file option is improper.')

    ############## adding a background from a file.
    if bkg_file!='None':
        bkg_file=os.path.abspath(args.bkg_file)
        b=h5py.File(bkg_file,'r')
        bkg=b['/entry_1/data/white_field'][x_min:x_max,y_min:y_max]
        b.close()
        geom = JUNGFRAUGeometry.from_crystfel_geom(geom_file)
        bkg, center = geom.position_modules_fast(bkg)
    elif bkg_file=='None':
        bkg = 0
    else:
        sys.exit('the mask file option is improper.')


    ##############


    HIT_counter=0
    HIT_frame_id_list=[]
    peakXPosRaw=np.zeros((len(frame_id_lst),1024))
    peakYPosRaw=np.zeros((len(frame_id_lst),1024))
    pixel_size=float(75e-6)#to be changed
    nPeaks=np.zeros((len(frame_id_lst),),dtype=np.int16)
    peakTotalIntensity=np.zeros((len(frame_id_lst),1024))

    lf=open(output_log_file,'w',1)
    ef=open(output_evt_lst_file,'w',1)

    lf.write(f'run : {args.run:d}, worker_rank: {rank:}')
    lf.write('\n-----------')
    lf.write('\nthld: %d\nmin_pix: %d\nmax_pix: %d\nmin_peak: %d\nmask_file: %s\nbkg_file: %s\nregion: %s'%\
    (thld,min_pix,max_pix,min_peak,mask_file,bkg_file,region))
    lf.write('\n-----------')

    for event_no in range(len(frame_id_lst)):
        frame_id = frame_id_lst[event_no]

        train_img_dict = CBD_ut.read_train(proposal,args.run,frame_id,\
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
        #     label_filtered=label[(area>args.min_pix)*(area<5e8)*(aspect_ratio>2)]
        label_filtered = label[(area > min_pix) * (area < max_pix)]
        #     area_filtered=area[(area>args.min_pix)*(area<5e8)*(aspect_ratio>2)]
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


        if (peak_no>=args.min_peak) and (peak_no<=1024):
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
    print(f'run: {args.run:d}\n')
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
