import numpy as np


def peakfinder_data_coords(peaks):
    peaks_fs = np.array(peaks.fs)  # y
    peaks_ss = np.array(peaks.ss)  # x

    mod = (peaks_ss // 512).astype(int)
    ss = (peaks_ss % 512).astype(int)
    fs = peaks_fs.astype(int)
    return np.hstack([mod[:, None], ss[:, None], fs[:, None]])
