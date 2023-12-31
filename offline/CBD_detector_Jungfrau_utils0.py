"""
utilities for reading the JUNGFRAU detector frames and calibration etc.
facility: EuXFEL, SPB


utilities for reading the JUNGFRAU detector frames and calibration etc.
facility: EuXFEL, SPB

Contacts:
Chufeng Li                           chufeng.li@cfel.de
Wenhui Zhang                         wenhui.zhang@desy.de
Mansi Butola                         mansi.butola@desy.de
Ivan De Gennaro Aquino               ivan.de.gennaro.aquino@desy.de
Andrew Morgan                        morganaj@unimelb.edu.au
Nikolay Ivanov                       nikolay.ivanov@desy.de
"""
import os
import sys

# import matplotlib.pyplot as plt
import extra_data
import h5py
import matplotlib.pyplot as plt
import numpy as np
from extra_data import open_run

# from extra_data import stack_detector_data
from extra_geom import JUNGFRAUGeometry


def read_train(
    proposal,
    run_id,
    train_ind,
    geom_file="/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom",
    geom_assem="False",
    ROI=(0, 2400, 0, 2400),
):
    """
    read the JUNGFRAU detector data from a given train.
    currently, the P700000 run 39 data, there is only one frame in one train.
    """
    run = open_run(proposal, run=run_id, data="proc")

    sel_img = run.select(
        [
            ("SPB_IRDA_JF4M/DET/JNGFR*:daqOutput", "data.adc"),
            ("SPB_IRDA_JF4M/DET/JNGFR*:daqOutput", "data.gain"),
            ("SPB_IRDA_JF4M/DET/JNGFR*:daqOutput", "data.mask"),
        ]
    )

    no_trains = sel_img["SPB_IRDA_JF4M/DET/JNGFR01:daqOutput"]["data.adc"].shape[0]

    tid, train_data = sel_img.train_from_index(train_ind)
    module_data_adc = extra_data.stack_detector_data(
        train_data,
        "data.adc",
        axis=-3,
        modules=8,
        starts_at=1,
        pattern=r"/DET/JNGFR(\d+)",
    )
    module_data_gain = extra_data.stack_detector_data(
        train_data,
        "data.gain",
        axis=-3,
        modules=8,
        starts_at=1,
        pattern=r"SPB_IRDA_JF4M/DET/JNGFR(\d+)",
    )
    module_data_mask = extra_data.stack_detector_data(
        train_data,
        "data.mask",
        axis=-3,
        modules=8,
        starts_at=1,
        pattern=r"SPB_IRDA_JF4M/DET/JNGFR(\d+)",
    )
    adc = module_data_adc
    gain = module_data_gain
    mask = module_data_mask

    train_img_dict = {
        "run_id": run_id,
        "no_trains": no_trains,
        "train_index": train_ind,
        "train_id": tid,
        "module_data_adc": module_data_adc,
        "module_data_gain": module_data_gain,
        "module_data_mask": module_data_mask,
        "geometry_file": geom_file,
    }
    if geom_assem == "False":
        return train_img_dict
    if geom_assem == "True":
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

        # 3D slicing
        train_img_dict["adc_img"] = adc_img[:, ROI[0] : ROI[1], ROI[2] : ROI[3]]
        train_img_dict["gain_img"] = gain_img[:, ROI[0] : ROI[1], ROI[2] : ROI[3]]
        train_img_dict["mask_img"] = mask_img[:, ROI[0] : ROI[1], ROI[2] : ROI[3]]
        train_img_dict["ROI"] = ROI
        return train_img_dict

    raise Exception("check the geom_assem argument!")


def get_3d_stack_from_train_ind(
    proposal,
    run_id,
    train_ind_tuple=(0, 50, 1),
    geom_file="/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom",
    geom_assem="False",
    ROI=(0, 2400, 0, 2400),
):

    train_ind_arry = np.arange(
        train_ind_tuple[0], train_ind_tuple[1], train_ind_tuple[2]
    )
    # list is faster to read in the data (compared with previous ndarray concatenation)
    stack_arry_module_adc = []
    stack_arry_module_gain = []
    stack_arry_module_mask = []
    stack_arry_img_adc = []
    stack_arry_img_gain = []
    stack_arry_img_mask = []

    for m in range(train_ind_arry.shape[0]):
        train_ind = train_ind_arry[m]
        train_img_dict = read_train(
            proposal,
            run_id,
            train_ind,
            geom_file=geom_file,
            geom_assem=geom_assem,
            ROI=ROI,
        )
        stack_arry_module_adc.append(train_img_dict["module_data_adc"])
        stack_arry_module_gain.append(train_img_dict["module_data_gain"])
        stack_arry_module_mask.append(train_img_dict["module_data_mask"])
        if geom_assem == "True":
            stack_arry_img_adc.append(train_img_dict["adc_img"])
            stack_arry_img_gain.append(train_img_dict["gain_img"])
            stack_arry_img_mask.append(train_img_dict["mask_img"])

    stack_arry_dict = {}
    stack_arry_dict["stack_arry_module_adc"] = stack_arry_module_adc
    stack_arry_dict["stack_arry_module_gain"] = stack_arry_module_gain
    stack_arry_dict["stack_arry_module_mask"] = stack_arry_module_mask
    if geom_assem == "True":
        stack_arry_dict["stack_arry_img_adc"] = stack_arry_img_adc
        stack_arry_dict["stack_arry_img_gain"] = stack_arry_img_gain
        stack_arry_dict["stack_arry_img_mask"] = stack_arry_img_mask

    return stack_arry_dict


def JungFrau_mask_maker(assembled_data, thres_mean=60, thres_sigma=2):
    data = assembled_data
    data_mean = np.mean(data, axis=0)
    data_sigma = np.std(data, axis=0)

    mask = np.zeros(data.shape[1:])

    mask[np.where(data_mean > thres_mean)] = 1
    mask[np.where(data_sigma > thres_sigma)] = 1
    mask_nan = np.logical_not(np.isnan(data_mean))
    mask_inf = np.logical_not(np.isinf(data_mean))

    mask = np.logical_not(mask)
    mask = mask * mask_nan * mask_inf

    # mask = np.logical_not(mask)

    # output a h5 file
    file_name = "mask.h5"
    with h5py.File(file_name, "w") as df:
        df.create_dataset("/entry_1/goodpixels", data=mask.astype("bool"))
    return mask


def make_white_field_mask(
    proposal,
    run_id,
    thres_mean=1e4,
    thres_sigma=1e4,
    geom_file="/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom",
):
    train_img_dict = read_train(
        proposal,
        run_id,
        0,
        geom_file=geom_file,
        geom_assem="False",
        ROI=(0, 2400, 0, 2400),
    )
    no_trains = train_img_dict["no_trains"]
    stride = int(no_trains // 100)
    print(f"{no_trains:d}  trains")
    stack_arry_dict = get_3d_stack_from_train_ind(
        proposal,
        run_id,
        train_ind_tuple=(0, no_trains, stride),
        geom_file=geom_file,
        geom_assem="False",
        ROI=(0, 2400, 0, 2400),
    )

    stack_arry = np.array(stack_arry_dict["stack_arry_module_adc"])
    stack_arry = stack_arry[:, 0, :, :]
    data = stack_arry
    data_mean = np.mean(data, axis=0)
    data_median = np.median(data, axis=0)
    data_sigma = np.std(data, axis=0)

    mask = np.ones(data.shape[1:])

    mrg = 2
    mask[:, 255:258, :] = 0
    mask[:, :, 255:258] = 0
    mask[:, :, 513:516] = 0
    mask[:, :, 771:774] = 0

    mask[:, 0:mrg, :] = 0
    mask[:, -mrg:, :] = 0
    mask[:, :, 0:mrg] = 0
    mask[:, :, -mrg:] = 0

    mask[np.where(data_mean > thres_mean)] = 0
    mask[np.where(data_sigma > thres_sigma)] = 0
    mask_nan = np.logical_not(np.isnan(data_mean))
    mask_inf = np.logical_not(np.isinf(data_mean))
    mask = mask * mask_nan * mask_inf

    with h5py.File(f"white_field_run_{run_id:d}.h5", "w") as wf:
        wf.create_dataset(
            "/entry_1/data/white_field", data=data_mean.astype(np.float64)
        )

    with h5py.File(f"median_white_field_run_{run_id:d}.h5", "w") as wf:
        wf.create_dataset(
            "/entry_1/data/median_white_field", data=data_median.astype(np.float64)
        )

    file_name = f"mask_run{run_id:d}.h5"
    with h5py.File(file_name, "w") as df:
        df.create_dataset("/entry_1/good_pixels", data=mask.astype("bool"))

    return
