'''
mask combination
CFL not tested yet.
2023-11-21 12:49:38

Usage:

python mask_comb.py <mask_file_1> .....<mask_file_n>
'''
import sys,os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
# import h5plugin

def find_h5paths(mask_file):
    mask_h5path = []
    possible_paths = [f'/entry_1/goodpixels',f'/entry_1/good_pixels',f'/data/mask',f'/mask/data',f'/data/data'] # can be modified
    with h5py.File(mask_file,'r') as m:
        for h5path_candidate in possible_paths:
            if h5path_candidate in m:
                mask_h5path.append(h5path_candidate)

    return mask_h5path


if __name__=='__main__':

    file_name_list=sys.argv[1:]
    file_name_list=[os.path.abspath(file_name) for file_name in file_name_list]

    out_file_lst = glob.glob('combined_mask_*.h5')
    if len(out_file_lst)==0:
        out_file_ind = 1
    else:
        out_file_ind_lst = [int(s.split('.')[0].split('_')[-1]) for s in out_file_lst]
        out_file_ind = int((np.array(out_file_ind_lst)).max()+1)
    out_file_name = f'combined_mask_{out_file_ind:d}.h5'

    for ind,file_name in enumerate(file_name_list):
        if ind==0:
            mask_h5path = find_h5paths(file_name)
            if len(mask_h5path)==0:
                sys.exit(f'check the mask h5 data path for: \n{file_name:s}')
            else:
                mask_h5path = mask_h5path[0]
                with h5py.File(file_name,'r') as m:
                    mask = np.array(m[mask_h5path]).astype(np.int8)
                mask_comb = mask
                MASK_SHAPE = mask.shape
        else:
            mask_h5path = find_h5paths(file_name)
            if len(mask_h5path)==0:
                sys.exit(f'check the mask h5 data path for: \n{file_name:s}')
            else:
                mask_h5path = mask_h5path[0]
                with h5py.File(file_name,'r') as m:
                    mask = np.array(m[mask_h5path]).astype(np.int8)
                if mask.shape!=MASK_SHAPE:
                    sys.exit(f'mask image shape not match, \n{file_name:s}')
                mask_comb *= mask
    mask_comb = mask_comb.astype('bool')
    with h5py.File(out_file_name,'w') as df:
        df.create_dataset('/entry_1/goodpixels',data=mask_comb)


    # plt.figure(figsize=(5,5))
    # plt.imshow(mask_comb,vmin=0,vmax=1)
    # plt.savefig(out_file_name[0:-3]+'.png')
