import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from skimage import measure, morphology, feature
from scipy.ndimage import convolve


def streak_evaluator(all_labels, ints_data, min_pix, area_max_size):
    """
    takes in thresholds and intensity data and returns all streaks
    found within.

    """
    props = measure.regionprops(all_labels, ints_data)
    # all_labels=measure.label(bimg_masked,connectivity=1) #connectivity is important here, for sim data,use 2, for exp data use 1

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
    print("area")
    print(area_filtered[area_sort_ind])
    return (
        label_filtered_sorted,
        weighted_centroid_filtered,
        props,
        all_labels,
    )


def single_streak_finder(
    img_array,
    thld=10,
    min_pix=15,
    mask=None,
    bkg=0,
    area_max_size=5e8,
):
    """
    takes in image data and finds all connected streaks above treshold and
    with a minimum area.

    """
    # start_time = time.time()
    if mask is None:
        mask = np.ones_like(img_array).astype(bool)

    img_array_bgd_subtracted = img_array - bkg
    bimg_masked = (img_array_bgd_subtracted > thld) * mask
    all_labels = measure.label(bimg_masked, connectivity=1)
    (
        label_filtered_sorted,
        weighted_centroid_filtered,
        props,
        all_labels,
    ) = streak_evaluator(all_labels, img_array_bgd_subtracted, min_pix, area_max_size)

    return (
        label_filtered_sorted,
        weighted_centroid_filtered,
        props,
        img_array,
        all_labels,
    )


def swell_array(arr, mode=1):
    assert mode in (1, 2)
    arr = np.array(arr, dtype=bool)
    # Define a kernel for convolution
    kernel4 = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
    )

    kernel8 = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    kernel = kernel4 if mode == 1 else kernel8

    # Perform 2D convolution with the defined kernel
    convolution_result = convolve(arr, kernel)

    # Set the values to 1 where convolution result is greater than 0
    result = np.where(convolution_result > 0, 1, 0)

    return result


def comb_streak_finder(
    img_array,
    thld=10,
    min_pix_indi=15,
    min_pix=15,
    mask=None,
    bkg=0,
    area_max_size=5e8,
    swell_N_times=1,
):
    """
    takes in image data and finds all connected streaks above treshold and
    with a minimum area -- allowing for individual streaks to be disconnected
    by upto N pixel.

    """
    # 1 find individual (sub-) streaks
    (label_filtered_sorted, _, _, _, all_labels) = single_streak_finder(
        img_array, thld, min_pix_indi, mask, bkg, area_max_size
    )

    # 2 find which streaks can be combined
    labels_indi = np.isin(all_labels, label_filtered_sorted)
    swolen_array = labels_indi.copy()
    for _ in range(swell_N_times):
        swolen_array = swell_array(swolen_array)

    img_array_bgd_subtracted = img_array - bkg
    swolen_labels = measure.label(swolen_array, connectivity=1)
    print("the following area count is inflated")
    (label_filtered_comb, _, _, all_labels_comb) = streak_evaluator(
        swolen_labels, img_array_bgd_subtracted, min_pix, area_max_size
    )

    # 3 adjust the single streak data to include disjointed streaks
    label_filtered_new = np.zeros(len(label_filtered_comb), dtype=int)

    for ii, label in enumerate(label_filtered_comb):
        label_select = all_labels_comb == label

        # occurence of all nonnegative integers (excluding 0)
        number_of_labels = np.bincount(all_labels[label_select])[1:]
        # choose the single_streak_finder label with the most pixels
        main_label = np.argmax(number_of_labels)
        label_filtered_new[ii] = main_label
        if np.sum(np.array(number_of_labels, dtype=bool)) == 1:
            continue
        label_select_exact = np.logical_and(label_select, all_labels > 0)
        all_labels[label_select_exact] = main_label

    (
        label_filtered_final,
        weighted_centroid_filtered,
        props,
        all_labels,
    ) = streak_evaluator(all_labels, img_array_bgd_subtracted, min_pix, area_max_size)

    return (
        label_filtered_final,
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
    upper_limit=30,
):
    plt.figure(figsize=(10, 10))
    plt.imshow(
        img_array * (mask.astype(np.int16)) if mask is not None else img_array,
        cmap="viridis",
        origin="upper",
    )
    plt.colorbar()
    # plt.clim(0,0.5*thld)
    plt.clim(0, upper_limit)
    # plt.xlim(250,2100)
    # plt.ylim(500,2300)

    for label in label_filtered_sorted:
        plt.scatter(props[label - 1].coords[:, 1], props[label - 1].coords[:, 0], s=0.5)
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
