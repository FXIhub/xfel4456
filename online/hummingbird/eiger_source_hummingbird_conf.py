import h5py
import hdf5plugin
import numpy as np
from hummingbird import analysis, plotting
from hummingbird.backend import add_record
from xfel4456.offline.hit_finding_utils import single_streak_finder

SCRATCH_DIR = "/gpfs/exfel/exp/SPB/202302/p004456/scratch/eiger16_test_files"
SCAN_ID = "Scan_335"

# streakogram parameters ---------------------
STREAKOGRAM_CONF_R_BIN_START, STREAKOGRAM_CONF_R_BIN_END = 0, 10_000
STREAKOGRAM_CONF_R_BINS = 1_000
STREAKOGRAM_CONF_I_BIN_START, STREAKOGRAM_CONF_I_BIN_END = 10, 100
STREAKOGRAM_CONF_I_BINS = 100

streakogram_conf = {
    "r_bins": np.linspace(
        STREAKOGRAM_CONF_R_BIN_START,
        STREAKOGRAM_CONF_R_BIN_END,
        STREAKOGRAM_CONF_R_BINS,
    ),
    "i_bins": np.linspace(
        STREAKOGRAM_CONF_I_BIN_START,
        STREAKOGRAM_CONF_I_BIN_END,
        STREAKOGRAM_CONF_I_BINS,
    ),
}

# powder plot parameters ---------------------
POWDER_CONF_X_BIN_START, POWDER_CONF_X_BIN_END = -2000, 2000
POWDER_CONF_X_BINS = 100
POWDER_CONF_Y_BIN_START, POWDER_CONF_Y_BIN_END = -2000, 2000
POWDER_CONF_Y_BINS = 100

powder_plot_conf = {
    "x_bins": np.linspace(
        POWDER_CONF_X_BIN_START,
        POWDER_CONF_X_BIN_END,
        POWDER_CONF_X_BINS,
    ),
    "y_bins": np.linspace(
        POWDER_CONF_Y_BIN_START,
        POWDER_CONF_Y_BIN_END,
        POWDER_CONF_Y_BINS,
    ),
}

# streak finding parameters ---------------------
MIN_STREAKS_THRESHOLD_FOR_HIT = 3
PIXELS_PERCENT_VALUE = 99.5
MIN_PIXELS_FOR_STREAK = 15

# load eiger file  ------------------------------
eiger_file = f"{SCRATCH_DIR}/4Mcropped_{SCAN_ID}_data_000001.h5"
print(f"loading eiger data {eiger_file}")
with h5py.File(eiger_file, "r") as eiger_file:
    eiger_data = np.array(eiger_file["entry/data/data"])
print("loaded eiger data")

# load mask -------------------------------------
mask = np.where(np.mean(eiger_data, axis=0) < 1e6, 1, 0)
# mask = np.ones(eiger_data[0].shape)
# for scan_93
mask[1130:1330, 1030:1250] = 0

# beam_center x,y position in pixels
beam_center = (1200, 1100)


def random_streaks_image():
    return eiger_data[np.random.randint(eiger_data.shape[0])]


def random_2dhist():
    data_shape = (50, 50)
    center = (12, 16)
    bg_data = np.random.random(data_shape)
    stronger = np.where(bg_data > 0.80, 0, 4)
    data = stronger + bg_data
    peaks = np.where(data > 4.78)
    distances = np.sqrt((peaks[0] - center[0]) ** 2 + (peaks[1] - center[1]) ** 2)
    intensity = data[peaks]
    distances_edges = np.linspace(min(distances), max(distances), num=10)
    intensity_edges = np.linspace(min(intensity), max(intensity), num=10)

    h, xedges, yedges = np.histogram2d(
        distances, intensity, bins=(distances_edges, intensity_edges)
    )
    h = h.T
    x, y = np.meshgrid(xedges, yedges)
    return h, y, x


def powder_hist(centroids, center, powder_plot_conf_dict):
    x_s = [(w[0] - center[0]) for w in centroids]
    y_s = [(w[1] - center[1]) for w in centroids]

    h, xedges, yedges = np.histogram2d(
        x_s,
        y_s,
        bins=(powder_plot_conf_dict["x_bins"], powder_plot_conf_dict["y_bins"]),
    )
    h = h.T
    x, y = np.meshgrid(xedges, yedges)
    return h, y, x


def eiger_frame_streaks_hist(
    streak_regions_labels,
    centroid_pos,
    props,
    center,
    streakogram_conf_dict,
    pick_intensity_max_or_mean="max",
):
    distances = [
        np.sqrt((w[0] - center[0]) ** 2 + (w[1] - center[1]) ** 2)
        for w in centroid_pos
    ]
    intensity = [
        props[l - 1].intensity_max
        if pick_intensity_max_or_mean == "max"
        else props[l - 1].intensity_mean * props[l - 1].area
        for l in streak_regions_labels
    ]

    h, xedges, yedges = np.histogram2d(
        distances,
        intensity,
        bins=(streakogram_conf_dict["r_bins"], streakogram_conf_dict["i_bins"]),
    )
    h = h.T
    x, y = np.meshgrid(xedges, yedges)
    return h, y, x


# Set new random seed
np.random.seed()

# Specify the facility
state = {}
state["Facility"] = "Dummy"

# Create a dummy facility
state["Dummy"] = {
    # The event repetition rate of the dummy facility [Hz]
    "Repetition Rate": 5,
    # Dictionary of data sources
    "Data Sources": {
        # The name of the data source.
        "CCD": {
            # A function that will generate the data for every event
            "data": lambda: random_streaks_image(),
            # The units to be used
            "unit": "ADU",
            # The name of the category for this data source.
            # All data sources are aggregated by type, which is the key
            # used when asking for them in the analysis code.
            "type": "photonPixelDetectors",
        }
    },
}


# This function is called for every single event
# following the given recipy of analysis
def onEvent(evt):

    # Processin rate [Hz]
    analysis.event.printProcessingRate()

    image_data = np.array(evt["photonPixelDetectors"]["CCD"].data)

    percentile_threshold = np.percentile(
        (mask * image_data).reshape(
            -1,
        ),
        PIXELS_PERCENT_VALUE,
    )

    (label_filtered_sorted, centroids, props, _, _,) = single_streak_finder(
        img_array=image_data * mask,
        thld=percentile_threshold,
        min_pix=MIN_PIXELS_FOR_STREAK,
    )

    # number of streaks history
    num_streaks_record = add_record(
        evt["analysis"],
        "analysis",
        "Number of streaks",
        len(centroids),
    )
    plotting.line.plotHistory(
        num_streaks_record,
        label="Number of streaks",
        hline=MIN_STREAKS_THRESHOLD_FOR_HIT,
    )

    if len(centroids) > MIN_STREAKS_THRESHOLD_FOR_HIT:
        # we have enough streaks
        evt["analysis"]["isStreak"] = True

        streak_hit = add_record(
            evt["analysis"],
            "analysis",
            "Streak Hit (raw)",
            image_data,
        )
        masked_streak_hit = add_record(
            evt["analysis"],
            "analysis",
            "Streak Hit with mask",
            image_data * mask,
        )
        masked_streak_hit_cumulative = add_record(
            evt["analysis"],
            "analysis",
            "Streak Hit with mask (cumulative)",
            image_data * mask,
        )
        plotting.image.plotImage(streak_hit, group="Hits", history=100)
        plotting.image.plotImage(masked_streak_hit, group="Hits", history=100)
        plotting.image.plotImage(
            masked_streak_hit_cumulative, group="Hits", history=100, sum_over=True
        )

        # streakogram histogram
        hist2d_streaks_data, _, _ = eiger_frame_streaks_hist(
            label_filtered_sorted,
            centroids,
            props,
            beam_center,
            streakogram_conf,
        )
        hist2d_streaks_record = add_record(
            evt["analysis"],
            "analysis",
            "streakogram",
            hist2d_streaks_data,
        )
        plotting.image.plotImage(
            hist2d_streaks_record,
            group="Hits",
            msg=f"X] radius: {STREAKOGRAM_CONF_R_BIN_START}:{STREAKOGRAM_CONF_R_BIN_END} pixels ({STREAKOGRAM_CONF_R_BINS} bins)\n"
            + f"Y] intensity: {STREAKOGRAM_CONF_I_BIN_START}:{STREAKOGRAM_CONF_I_BIN_END} photons ({STREAKOGRAM_CONF_I_BINS} bins)",
            aspect_ratio=STREAKOGRAM_CONF_I_BINS / STREAKOGRAM_CONF_R_BINS,
            history=1000,
            sum_over=True,
        )

        # powder histogram
        powder_hist2d, _, _ = powder_hist(
            centroids,
            beam_center,
            powder_plot_conf,
        )
        powder_hist2d_record = add_record(
            evt["analysis"],
            "analysis",
            "powder",
            powder_hist2d,
        )
        plotting.image.plotImage(
            powder_hist2d_record,
            group="Hits",
            msg=f"X] x_pos: {POWDER_CONF_X_BIN_START}:{POWDER_CONF_X_BIN_END} pixels ({POWDER_CONF_X_BINS} bins)\n"
            + f"Y] y_pos: {POWDER_CONF_Y_BIN_START}:{POWDER_CONF_Y_BIN_END} pixels ({POWDER_CONF_Y_BINS} bins)",
            history=1000,
            sum_over=True,
        )

    else:
        evt["analysis"]["isStreak"] = False

        not_a_streak_hit = add_record(
            evt["analysis"],
            "analysis",
            "No Streak (raw)",
            image_data,
        )
        masked_not_a_streak_hit = add_record(
            evt["analysis"],
            "analysis",
            "No Streak with mask",
            image_data * mask,
        )
        masked_not_a_streak_hit_cumulative = add_record(
            evt["analysis"],
            "analysis",
            "No Streak with mask (cumulative)",
            image_data * mask,
        )
        plotting.image.plotImage(not_a_streak_hit, group="Non Hits", history=100)
        plotting.image.plotImage(masked_not_a_streak_hit, group="Non Hits", history=100)
        plotting.image.plotImage(
            masked_not_a_streak_hit_cumulative,
            group="Non Hits",
            history=100,
            sum_over=True,
        )

    # hitrate
    analysis.hitfinding.hitrate(
        evt,
        evt["analysis"]["isStreak"],
        history=1000,
    )
    plotting.line.plotHistory(evt["analysis"]["hitrate"], label="Hit rate [%]")
