import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from skimage import measure, morphology, feature


def single_streak_finder(
    img_array,
    thld=10,
    min_pix=15,
    mask = None,
    bkg = 0,
    area_max_size=5e8,
):
    # start_time = time.time()
    if mask is None : 
        mask = np.ones_like(img_array).astype(bool)
    
    img_array_bgd_subtracted = img_array - bkg
    bimg_masked = (img_array_bgd_subtracted > thld) * mask

    all_labels = measure.label(bimg_masked, connectivity=1)
    # all_labels=measure.label(bimg_masked,connectivity=1) #connectivity is important here, for sim data,use 2, for exp data use 1
    props = measure.regionprops(all_labels, img_array_bgd_subtracted)

    area = np.array([r.area for r in props]).reshape(
        -1,
    )
    # max_intensity = np.array([r.max_intensity for r in props]).reshape(-1, )

    # major_axis_length = np.array([r.major_axis_length for r in props]).reshape(-1, )
    # minor_axis_length = np.array([r.minor_axis_length for r in props]).reshape(-1, )
    # aspect_ratio = major_axis_length / (minor_axis_length + 1)
    # coords=np.array([r.coords for r in props]).reshape(-1,)
    
    label = np.array([r.label for r in props]).reshape(
        -1,
    )
  
    # centroid = np.array([np.array(r.centroid).reshape(1, 2) for r in props]).reshape((-1, 2))
    # weighted_centroid = np.array([r.weighted_centroid for r in props]).reshape(-1, )
    #     label_filtered=label[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
    
    label_filtered = label[(area > min_pix) * (area < area_max_size)]
    #     area_filtered=area[(area>min_pix)*(area<5e8)*(aspect_ratio>2)]
    area_filtered = area[(area > min_pix) * (area < area_max_size)]
    area_sort_ind = np.argsort(area_filtered)[::-1]
    label_filtered_sorted = label_filtered[area_sort_ind]
    
    # area_filtered_sorted = area_filtered[area_sort_ind]

    # weighted_centroid_filtered_start_time = time.time()

    weighted_centroid_filtered = np.zeros((len(label_filtered_sorted), 2))
    for index, value in enumerate(label_filtered_sorted):
        weighted_centroid_filtered[index, :] = np.array(
            props[value - 1].weighted_centroid
        )
      
    # print('In image: %s \n %5d peaks are found' %(img_file_name, len(label_filtered_sorted)))
    # beam_center=np.array([1492.98,2163.41])
    
    # weighted_centroid_filtered_end_timee = time.time()
    # print(f"weight timings {weighted_centroid_filtered_end_timee-weighted_centroid_filtered_start_time:.10f}")
    
    # end_time = time.time()
    # print(f"full timing {end_time - start_time}")
    return (
        label_filtered_sorted,
        weighted_centroid_filtered,
        props,
        img_array,
        all_labels,
    )

def plot_streaks(
        label_filtered_sorted,
        weighted_centroid_filtered,
        props,
        img_array,
        all_labels,
        mask=None,
        interacting=False,
        fig_filename="streak_finding.png",
    ):

    plt.figure(figsize=(15, 15))
    plt.imshow(img_array * (mask.astype(np.int16)) if mask is not None else img_array, cmap="viridis", origin="upper")
    plt.colorbar()
    # plt.clim(0,0.5*thld)
    plt.clim(0, 30)
    # plt.xlim(250,2100)
    # plt.ylim(500,2300)

    for label in label_filtered_sorted:
        plt.scatter(
            props[label - 1].coords[:, 1], props[label - 1].coords[:, 0], s=0.5
        )
    plt.scatter(
        weighted_centroid_filtered[:, 1],
        weighted_centroid_filtered[:, 0],
        edgecolors="r",
        facecolors="none",
    )

    #    plt.scatter(beam_center[1],beam_center[0],marker='*',color='b')
    #         title_Str=exp_img_file+'\nEvent: %d '%(frame_no)
    #         plt.title(title_Str)
    if interacting:
        plt.show()
    else:
        plt.savefig(fig_filename)

