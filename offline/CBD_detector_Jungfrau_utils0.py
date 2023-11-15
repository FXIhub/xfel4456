'''
utilities for reading the JUNGFRAU detector frames and calibration etc.
facility: EuXFEL, SPB

'''

from extra_data import open_run
import h5py
import numpy as np
import matplotlib.pyplot as plt
import extra_data
from extra_data import stack_detector_data
from extra_geom import JUNGFRAUGeometry

from skimage import measure, morphology, feature
import scipy
import glob

def read_train(proposal,run_id,train_ind,\
geom_file='/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom',geom_assem='False'):
    '''
    read the JUNGFRAU detector data from a given train.
    currently, the P700000 run 39 data, there is only one frame in one train.
    '''
    run = open_run(proposal, run = run_id,data='proc')

    sel_img = run.select([('SPB_IRDA_JF4M/DET/JNGFR*:daqOutput', 'data.adc'),\
                            ('SPB_IRDA_JF4M/DET/JNGFR*:daqOutput', 'data.gain'),\
                            ('SPB_IRDA_JF4M/DET/JNGFR*:daqOutput', 'data.mask')])
    tid, train_data = sel_img.train_from_index(train_ind)
    module_data_adc =extra_data.stack_detector_data(train_data,'data.adc',axis=-3,modules=8,starts_at=1,pattern=r'/DET/JNGFR(\d+)')
    module_data_gain =extra_data.stack_detector_data(train_data,'data.gain',axis=-3,modules=8,starts_at=1,pattern=r'SPB_IRDA_JF4M/DET/JNGFR(\d+)')
    module_data_mask =extra_data.stack_detector_data(train_data,'data.mask',axis=-3,modules=8,starts_at=1,pattern=r'SPB_IRDA_JF4M/DET/JNGFR(\d+)')
    adc = module_data_adc
    gain = module_data_gain
    mask = module_data_mask

    train_img_dict=\
    {'run_id':run_id,'train_index':train_ind,'train_id':tid,\
    'module_data_adc':module_data_adc,'module_data_gain':module_data_gain,\
    'module_data_mask':module_data_mask,'geometry_file':geom_file}
    if geom_assem=='False':
        return train_img_dict
    elif geom_assem=='True':
        geom = JUNGFRAUGeometry.from_crystfel_geom(geom_file)

        adc_img, center = geom.position_modules(adc)
        adc_img[np.where(np.isnan(adc_img))] = 0
        adc_img[np.where(np.isinf(adc_img))] = 0

        gain_img, center = geom.position_modules(gain)
        gain_img[np.where(np.isnan(gain_img))] = 0
        gain_img[np.where(np.isinf(gain_img))] = 0

        mask_img, center = geom.position_modules(mask)
        mask_img[np.where(np.isnan(mask_img))] = 0
        mask_img[np.where(np.isinf(mask_img))] = 0

        train_img_dict['adc_img'] = adc_img
        train_img_dict['gain_img'] = gain_img
        train_img_dict['mask_img'] = mask_img
        return train_img_dict
    else:
        sys.error('check the geom_assem argument!')
        return None


def single_peak_finder(img_arry,thld=10,min_pix=15,mask_file='None',interact=False):

    if mask_file!='None':
        mask_file=os.path.abspath(mask_file)
        m=h5py.File(mask_file,'r')
        mask=np.array(m['/data/data']).astype(bool)
        bkg = np.array(m['/data/bkg'])
        m.close()
    elif mask_file=='None':
        mask=np.ones_like(img_arry).astype(bool)
        bkg = 0
    else:
        sys.exit('the mask file option is inproper.')


    img_arry = img_arry - bkg
    bimg = (img_arry>thld)
    bimg=bimg*mask


    all_labels=measure.label(bimg,connectivity=1)
    #all_labels=measure.label(bimg,connectivity=1) #connectivity is important here, for sim data,use 2, for exp data use 1
    props=measure.regionprops(all_labels,img_arry)

    area=np.array([r.area for r in props]).reshape(-1,)
    max_intensity=np.array([r.max_intensity for r in props]).reshape(-1,)

    major_axis_length=np.array([r.major_axis_length for r in props]).reshape(-1,)
    minor_axis_length=np.array([r.minor_axis_length for r in props]).reshape(-1,)
    aspect_ratio = major_axis_length/(minor_axis_length+1)
    #coords=np.array([r.coords for r in props]).reshape(-1,)
    label=np.array([r.label for r in props]).reshape(-1,)
    centroid=np.array([np.array(r.centroid).reshape(1,2) for r in props]).reshape((-1,2))
    weighted_centroid=np.array([r.weighted_centroid for r in props]).reshape(-1,)
#     label_filtered=label[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
    label_filtered=label[(area>min_pix)*(area<5e8)]
#     area_filtered=area[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
    area_filtered=area[(area>min_pix)*(area<5e8)]
    area_sort_ind=np.argsort(area_filtered)[::-1]
    label_filtered_sorted=label_filtered[area_sort_ind]
    area_filtered_sorted=area_filtered[area_sort_ind]
    weighted_centroid_filtered=np.zeros((len(label_filtered_sorted),2))
    for index,value in enumerate(label_filtered_sorted):

            weighted_centroid_filtered[index,:]=np.array(props[value-1].weighted_centroid)
#    print('In image: %s \n %5d peaks are found' %(img_file_name, len(label_filtered_sorted)))
    #beam_center=np.array([1492.98,2163.41])

    if interact:
        plt.figure(figsize=(15,15))
        plt.imshow(img_arry*(mask.astype(np.int16)),cmap='viridis',origin='upper')
        plt.colorbar()
    #    plt.clim(0,0.5*thld)
        plt.clim(0,30)
        #plt.xlim(250,2100)
        #plt.ylim(500,2300)
        plt.scatter(weighted_centroid_filtered[:,1],weighted_centroid_filtered[:,0],edgecolors='r',facecolors='none')
        for label in label_filtered_sorted:
            plt.scatter(props[label-1].coords[:,1],props[label-1].coords[:,0],s=0.5)
    #    plt.scatter(beam_center[1],beam_center[0],marker='*',color='b')
#         title_Str=exp_img_file+'\nEvent: %d '%(frame_no)
#         plt.title(title_Str)
        plt.show()
    return label_filtered_sorted,weighted_centroid_filtered,props,img_arry,all_labels


def get_3d_stack_from_train_ind(proposal,run_id,train_ind_tuple=(0,50,1),\
geom_file='/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom',geom_assem='False'):

    train_ind_arry = np.arange(train_ind_tuple[0],train_ind_tuple[1],train_ind_tuple[2])

    for m in range(train_ind_arry.shape[0]):
        if m==0:
            train_ind = train_ind_arry[m]
            train_img_dict = read_train(proposal,run_id,train_ind,geom_file=geom_file,geom_assem=geom_assem)
            stack_arry_module_adc = train_img_dict['module_data_adc']
            stack_arry_module_gain = train_img_dict['module_data_gain']
            stack_arry_module_mask = train_img_dict['module_data_mask']
            if geom_assem=='True':
                stack_arry_img_adc = train_img_dict['adc_img']
                stack_arry_img_gain = train_img_dict['gain_img']
                stack_arry_img_mask = train_img_dict['mask_img']
        else:
            train_ind = train_ind_arry[m]
            train_img_dict = read_train(proposal,run_id,train_ind,geom_file=geom_file,geom_assem=geom_assem)
            stack_arry_module_adc = np.concatenate((stack_arry_module_adc,train_img_dict['module_data_adc']),axis=0)
            stack_arry_module_gain = np.concatenate((stack_arry_module_gain,train_img_dict['module_data_gain']),axis=0)
            stack_arry_module_mask = np.concatenate((stack_arry_module_mask,train_img_dict['module_data_mask']),axis=0)
            if geom_assem=='True':
                stack_arry_img_adc = np.concatenate((stack_arry_img_adc,train_img_dict['adc_img']),axis=0)
                stack_arry_img_gain = np.concatenate((stack_arry_img_gain,train_img_dict['gain_img']),axis=0)
                stack_arry_img_mask = np.concatenate((stack_arry_img_mask,train_img_dict['mask_img']),axis=0)

    stack_arry_dict = dict()
    stack_arry_dict['stack_arry_module_adc'] = stack_arry_module_adc
    stack_arry_dict['stack_arry_module_gain'] = stack_arry_module_gain
    stack_arry_dict['stack_arry_module_mask'] = stack_arry_module_mask
    if geom_assem=='True':
        stack_arry_dict['stack_arry_img_adc'] = stack_arry_img_adc
        stack_arry_dict['stack_arry_img_gain'] = stack_arry_img_gain
        stack_arry_dict['stack_arry_img_mask'] = stack_arry_img_mask


    return stack_arry_dict
# img_arry = adc_img
# label_filtered_sorted,weighted_centroid_filtered,props,img_arry,all_labels = \
# single_peak_finder(img_arry,thld=10,min_pix=5,mask_file='None',interact=True)
# print(len(label_filtered_sorted))



def JungFrau_mask_maker(assembled_data,thres_mean=60,thres_sigma=2):
    
        data = assembled_data
        data_mean = np.mean(data,axis=0)
        data_sigma = np.std(data,axis=0)
        
        mask = np.zeros(data.shape[1:])
    
        mask[np.where(data_mean>thres_mean)] = 1
        mask[np.where(data_sigma>thres_sigma)] = 1
        mask_nan = np.logical_not(np.isnan(data_mean))
        
        
        mask = np.logical_not(mask)
        mask = mask*mask_nan
        
        #mask = np.logical_not(mask)
        
        stack_arry_dict['stack_arry_img_mask_updated'] = mask
        
        return mask
