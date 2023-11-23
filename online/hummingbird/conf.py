import sys
import os
import time
import numpy as np
import h5py
import extra_geom
import warnings
from scipy.optimize import curve_fit

from hummingbird import plotting
from hummingbird import analysis
from hummingbird import ipc
from hummingbird.backend import add_record

prop_dir = "/gpfs/exfel/exp/SPB/202302/p004456"
sys.path.append(os.path.join(prop_dir, "usr/Shared/hummingbird/utils"))

from draw import PseudoPowderDiffraction, Peakogram


state = {}
state["Facility"] = "EuXFEL"
state["EventIsTrain"] = True
state["EuXFEL/DataSource"] = "tcp://exflong102-ib:55555"
state["EuXFEL/DataFormat"] = "Calib"
state["EuXFEL/SelModule"] = None
state["EuXFEL/MaxTrainAge"] = 4e20
state["EuXFEL/FirstCell"] = 0
state["EuXFEL/LastCell"] = 15

# jf1m_offsets = [(95.0, 564.0), (95.0, 17.0)]
# jf1m_orient = [(-1, -1), (-1, -1)]
# geom = extra_geom.JUNGFRAUGeometry.from_module_positions(jf1m_offsets, jf1m_orient)

# geometry ----------------------------------------------------------

geom = extra_geom.JUNGFRAUGeometry.from_crystfel_geom(
    os.path.join(prop_dir, "usr/geometry/geom_v5.geom")
)
nmod = geom.n_modules
xyz = geom.get_pixel_positions()

adu_per_photon = 7.4
clen = 0.55
wavelength = 10.33e-10


# streak finder parameters ---------------------------------------------

streak_finder_thr_percent = 99.5
streak_finder_min_pixels = 20
streak_finder_max_area = 800
streak_finder_num_streaks_threshold = 5

# masks ---------------------------------------------------------------

mrg = 2  # borders of the modules are too bright

user_mask = np.ones(geom.expected_data_shape, bool)
# user_mask[:, :mrg, :] = False
# user_mask[:, -mrg:, :] = False
# user_mask[:, :, :mrg] = False
# user_mask[:, :, -mrg:] = False

mask_file = f"{prop_dir}/usr/Shared/hummingbird/hummingbird_current_mask.h5"
with h5py.File(mask_file, "r") as maskh5:
    mask_h5 = np.array(maskh5["entry_1/goodpixels"])

ring_mask_file = f"{prop_dir}/usr/Shared/hummingbird/mask_combined_with_ring_r0085.h5"
with h5py.File(ring_mask_file, "r") as maskh5:
    ring_mask_h5 = np.array(maskh5["entry_1/goodpixels"])

mask_h5 = ring_mask_h5 & mask_h5

# background whitefield file (dividing) ------------------------------------------

whitefield_file = f"{prop_dir}/usr/Shared/hummingbird/white_field_run_99.h5"
with h5py.File(whitefield_file, "r") as bgh5:
    background_data = np.array(bgh5["entry_1/data/white_field"])

# zeroing this for now
# background_data = np.zeros(geom.expected_data_shape, float)

# median_whitefield_file = (
#     f"{prop_dir}/usr/Shared/hummingbird/median_white_field_run_99.h5"
# )
# with h5py.File(median_whitefield_file, "r") as bgh5:
#     median_bgd = np.array(bgh5["entry_1/data/median_white_field"])

# running white field background ------------------------------------------------

running_wfs = [np.ones(geom.expected_data_shape, float)]
length_running_white_fields = 1000

# operations ------------------------------------------------------------------

send_hits = True
send_powdersum = True
send_profiles = False
do_peakfinding = False
do_streakfinding = True

apply_running_background_division = False
apply_background_division = False
apply_background_subtraction = True


if do_streakfinding:
    from streakfinder import StreakFinder

    streak_finder = StreakFinder(
        thr_percent=streak_finder_thr_percent,
        min_pixels=streak_finder_min_pixels,
        max_area=streak_finder_max_area,
    )
    num_streaks_threshold = streak_finder_num_streaks_threshold

    streakogram = Peakogram(
        geom, nr=500, rmin=0, rmax=1500, nintens=200, intens_max=200
    )
    streaks_powder = PseudoPowderDiffraction(geom)


if do_peakfinding:
    from cfelpyutils.peakfinding import Peakfinder8PeakDetection
    from peakfinder import peakfinder_data_coords

    r = (np.sqrt(np.sum(xyz * xyz, 3)).reshape(-1, 512) / geom.pixel_size).astype(
        np.float32
    )

    peak_finder = Peakfinder8PeakDetection(
        max_num_peaks=1000,
        asic_ny=256,
        asic_nx=256,
        nasics_y=4 * geom.n_modules,
        nasics_x=2,
        adc_threshold=10,
        minimum_snr=6,
        min_pixel_count=3,
        max_pixel_count=10,
        local_bg_radius=6,
        min_res=30,
        max_res=700,
        bad_pixel_map_filename=None,
        bad_pixel_map_hdf5_path=None,
        radius_pixel_map=r,
    )
    num_peaks_threshold = 10

    peakogram = Peakogram(geom, nr=500, rmin=0, rmax=1500, nintens=200, intens_max=200)
    peaks_powder = PseudoPowderDiffraction(geom)


if send_profiles:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    ai = AzimuthalIntegrator(
        detector=geom.to_pyfai_detector(),
        dist=clen,  # Sample-detector distance (m)
        wavelength=wavelength,  # Wavelength (m)
    )


def onEvent(evt):
    sys.stdout.flush()
    analysis.event.printProcessingRate()

    det = evt["photonPixelDetectors"]["JF4M Stacked"]

    det_data = np.moveaxis(det.data[:, 0], (1, 2), (2, 1))

    if apply_running_background_division:
        if len(running_wfs) > length_running_white_fields:
            running_wfs.pop(0)
        running_wfs.append(det_data)

        running_white_field = np.mean(running_wfs, axis=0, dtype=float)
        running_white_field[~np.isfinite(running_white_field)] = 1
        running_white_field[background_data <= 1] = 1
        det_data = det_data / running_white_field

    if apply_background_division:
        background_data[~np.isfinite(background_data)] = 1
        background_data[background_data <= 1] = 1
        det_data = det_data / background_data

    if apply_background_subtraction:
        background_data[~np.isfinite(background_data)] = 0
        background_data[background_data <= 1] = 0
        correction_factor = np.nansum(det_data) / np.nansum(background_data)
        det_data = det_data - correction_factor * background_data

    if not any(
        [
            apply_background_division,
            apply_background_subtraction,
            apply_running_background_division,
        ]
    ) or all(
        [
            apply_background_division,
            apply_background_subtraction,
            apply_running_background_division,
        ]
    ):
        raise Exception("wrong operation setup, one and only one should be true")

    msk = (np.isfinite(det_data) & user_mask) * mask_h5

    masked_data = det_data.copy()
    masked_data[~msk] = np.nan

    assem, center = geom.position_modules_fast(masked_data)
    assem[np.isnan(assem)] = -1

    if do_peakfinding:
        peak_finder._mask = msk.astype(np.int8).reshape(-1, 512)
        peaks = peak_finder.find_peaks(det_data.reshape(-1, 512))

        num_peaks_rec = add_record(
            evt["analysis"], "analysis", "numpeaks", peaks.num_peaks
        )
        plotting.line.plotHistory(
            num_peaks_rec, group="Hitfinding", hline=num_peaks_threshold, history=10000
        )

        if peaks.num_peaks > 0:
            a = peaks_powder.get(peakfinder_data_coords(peaks))
            peaks_powder_image = add_record(
                evt["analysis"], "analysis", "Peaks powder", a
            )
            plotting.image.plotImage(
                peaks_powder_image, group="Images", history=1, sum_over=True
            )

    if do_streakfinding:
        streaks = streak_finder.find_streaks(det_data, msk)

        num_streaks = add_record(
            evt["analysis"], "analysis", "numstreaks", streaks.num_streaks
        )
        plotting.line.plotHistory(
            num_streaks, group="Hitfinding", hline=num_streaks_threshold, history=10000
        )

        if streaks.num_streaks > num_streaks_threshold:
            # this is a hit
            a = streaks_powder.get(streaks.positions)
            streaks_powder_hit = add_record(
                evt["analysis"], "analysis", "Streaks powder (hits)", a
            )
            plotting.image.plotImage(
                streaks_powder_hit, group="Images", history=1, sum_over=True
            )

            b = streakogram.get(streaks.positions, streaks.intens[:, 0])
            streakogram_image = add_record(
                evt["analysis"], "analysis", "Streakogram", b
            )
            plotting.image.plotImage(
                streakogram_image, group="Images", history=1, sum_over=True
            )

            image_hit = add_record(evt["analysis"], "analysis", "Hit", assem)
            plotting.image.plotImage(image_hit, group="Images", history=10)

            integral_hit = add_record(
                evt["analysis"], "analysis", "Hit Integral", assem
            )
            plotting.image.plotImage(
                integral_hit, group="Images", history=1, sum_over=True
            )

        else:
            # this is a mis
            if streaks.num_streaks > 0:
                a = streaks_powder.get(streaks.positions)
                streaks_powder_mis = add_record(
                    evt["analysis"], "analysis", "Streaks powder (miss)", a
                )
                plotting.image.plotImage(
                    streaks_powder_mis, group="Images", history=1, sum_over=True
                )

            image_mis = add_record(evt["analysis"], "analysis", "Mis", assem)
            plotting.image.plotImage(image_mis, group="Images", history=10)

            integral_mis = add_record(
                evt["analysis"], "analysis", "Mis Integral", assem
            )
            plotting.image.plotImage(
                integral_mis, group="Images", history=1, sum_over=True
            )

        # hitrate
        evt["analysis"]["isHit"] = streaks.num_streaks > num_streaks_threshold
        analysis.hitfinding.hitrate(
            evt,
            evt["analysis"]["isHit"],
            history=1000,
        )
        if ipc.mpi.is_main_event_reader():
            plotting.line.plotHistory(
                evt["analysis"]["hitrate"], group="Hitfinding", label="Hit rate [%]"
            )
