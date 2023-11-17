import extra_data
import extra_geom
import h5py
import hdf5plugin
import numpy as np
from hummingbird import analysis, plotting
from hummingbird.backend import add_record
from xfel4456.offline.CBD_detector_Jungfrau_utils0 import read_train
from xfel4456.offline.hit_finding_utils import single_streak_finder

BEAMTIME_DIR = "/gpfs/exfel/exp/SPB/202302/p004456/"
PROP_ID = "700000"
RUN_ID = "0039"

# define geometry
GEOM_FILE = "/gpfs/exfel/exp/XMPL/201750/p700000/proc/r0040/j4m-p2805_v03.geom"
geom = extra_geom.JUNGFRAUGeometry.from_crystfel_geom(GEOM_FILE)
beam_center = (0.0, 0.0)  # assuming beam center in the center of the detector geom

# this run
THIS_RUN = extra_data.open_run(PROP_ID, RUN_ID, "proc")
THIS_RUN.info()


def get_jungfrau_data_by_random_train_index(
    run_id=RUN_ID, prop_id=PROP_ID, geometry_file=GEOM_FILE
):
    rnd_train_index = np.random.randint(len(THIS_RUN.train_ids))
    train_data = read_train(
        proposal=prop_id,
        run_id=run_id,
        train_ind=rnd_train_index,
        geom_assem="True",
        geom_file=geometry_file,
    )
#     if rnd_train_index % 3:
#         # some random streaks
#         train_data["module_data_adc"][0][0][100:120, 50:60] = 50
#         train_data["adc_img"][0][100:120, 50:60] = 100

#         train_data["module_data_adc"][0][0][200:220, 95:105] = 75
#         train_data["adc_img"][0][200:220, 95:105] = 100

#         train_data["module_data_adc"][0][0][600:620, 450:470] = 100
#         train_data["adc_img"][0][600:620, 450:470] = 100
    return train_data


# streakogram parameters ---------------------
STREAKOGRAM_CONF_R_BIN_START, STREAKOGRAM_CONF_R_BIN_END = 0, 1
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
POWDER_CONF_X_BIN_START, POWDER_CONF_X_BIN_END = -0.2, 0.2
POWDER_CONF_X_BINS = 200
POWDER_CONF_Y_BIN_START, POWDER_CONF_Y_BIN_END = -0.2, 0.2
POWDER_CONF_Y_BINS = 200

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
MIN_PIXELS_FOR_STREAK = 5
STREAK_AREA_MAX = 2000

# load mask -------------------------------------
# mask ...


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def jungfrau_powder_hist(
    streaks_per_module_dict,
    jungfrau_geom,
    center,
    powder_plot_conf_dict,
):
    x_list = []
    y_list = []
    for key, val in streaks_per_module_dict.items():
        y_this_module = [
            (
                jungfrau_geom.get_pixel_positions()[key][
                    int(centroid[0]), int(centroid[1])
                ][0]
                - center[0]
            )
            for centroid in val["centroids"]
        ]
        x_this_module = [
            (
                jungfrau_geom.get_pixel_positions()[key][
                    int(centroid[0]), int(centroid[1])
                ][1]
                - center[1]
            )
            for centroid in val["centroids"]
        ]
        x_list.append(x_this_module)
        y_list.append(y_this_module)

    x_s = np.array(sum(x_list, []))
    y_s = np.array(sum(y_list, []))

    print(f"{min(x_s)=}, {max(x_s)=}")
    print(f"{min(y_s)=}, {max(y_s)=}")

    h, x_bins_edges, y_bins_edges = np.histogram2d(
        x_s,
        y_s,
        bins=(powder_plot_conf_dict["x_bins"], powder_plot_conf_dict["y_bins"]),
    )
    h = h.T
    x, y = np.meshgrid(x_bins_edges, y_bins_edges)
    return h, y, x


def jungfrau_frame_streaks_hist(
    streaks_per_module_dict,
    jungfrau_geom,
    center,
    streakogram_conf_dict,
    pick_intensity_max_or_mean="max",
):
    dist_list = []
    ints_list = []
    for key, val in streaks_per_module_dict.items():
        dist_this_module = [
            np.sqrt(
                (
                    jungfrau_geom.get_pixel_positions()[key][
                        int(centroid[0]), int(centroid[1])
                    ][0]
                    - center[0]
                )
                ** 2
                + (
                    jungfrau_geom.get_pixel_positions()[key][
                        int(centroid[0]), int(centroid[1])
                    ][1]
                    - center[1]
                )
                ** 2
            )
            for centroid in val["centroids"]
        ]
        ints_this_module = [
            val["props"][l - 1].intensity_max
            if pick_intensity_max_or_mean == "max"
            else val["props"][l - 1].intensity_mean * val["props"][l - 1].area
            for l in val["labels"]
        ]
        dist_list.append(dist_this_module)
        ints_list.append(ints_this_module)

    distances = np.array(sum(dist_list, []))
    intensities = np.array(sum(ints_list, []))

    print(f"{min(distances)=}, {max(distances)=}")
    print(f"{min(intensities)=}, {max(intensities)=}")

    h, xedges, yedges = np.histogram2d(
        distances,
        intensities,
        bins=(streakogram_conf_dict["r_bins"], streakogram_conf_dict["i_bins"]),
    )
    h = h.T
    x, y = np.meshgrid(xedges, yedges)
    return h, y, x


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
            "data": lambda: get_jungfrau_data_by_random_train_index(),
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

    data = evt["photonPixelDetectors"]["CCD"].data

    modules_data_adc = data["module_data_adc"][0]
    modules_data_mask = data["module_data_mask"][0]
    masked_modules_data = modules_data_adc

    img_adc = data["adc_img"][0]
    img_mask = data["mask_img"][0]
    masked_img = img_adc

    percentile_threshold = np.percentile(
        (masked_img).reshape(-1),
        PIXELS_PERCENT_VALUE,
    )
    print(f"{percentile_threshold=}")

    streaks_per_module_dict = {}
    for i in range(masked_modules_data.shape[0]):
        # print(f"module {i}")
        (labels, centroids, props, _, _,) = single_streak_finder(
            img_array=masked_modules_data[i],
            thld=percentile_threshold,
            min_pix=MIN_PIXELS_FOR_STREAK,
            area_max_size=STREAK_AREA_MAX,
        )

        labels_nan_removed = []
        centroids_nan_removed = []
        for c_i, c in enumerate(centroids):
            if not any(np.isnan(x) for x in c):
                centroids_nan_removed.append(c)
                labels_nan_removed.append(labels[c_i])

        # print(f"{centroids_nan_removed=}, {labels_nan_removed=}")
        streaks_per_module_dict[i] = {
            "labels": labels_nan_removed,
            "centroids": centroids_nan_removed,
            "props": props,
        }

    def num_of_streaks():
        return sum(
            [len(val["centroids"]) for key, val in streaks_per_module_dict.items()]
        )

    def enough_streaks_for_a_hit():
        return MIN_STREAKS_THRESHOLD_FOR_HIT < num_of_streaks()

    # number of streaks history
    num_streaks_record = add_record(
        evt["analysis"],
        "analysis",
        "Number of streaks",
        num_of_streaks(),
    )
    plotting.line.plotHistory(
        num_streaks_record,
        label="Number of streaks",
        hline=MIN_STREAKS_THRESHOLD_FOR_HIT,
    )

    if enough_streaks_for_a_hit():
        # we have enough streaks
        evt["analysis"]["isStreak"] = True

        masked_streak_hit = add_record(
            evt["analysis"],
            "analysis",
            "Streak Hit with mask",
            masked_img,
        )
        masked_streak_hit_cumulative = add_record(
            evt["analysis"],
            "analysis",
            "Streak Hit with mask (cumulative)",
            masked_img,
        )
        plotting.image.plotImage(masked_streak_hit, group="Hits", history=100)
        plotting.image.plotImage(
            masked_streak_hit_cumulative, group="Hits", history=100, sum_over=True
        )

        # streakogram histogram

        hist2d_streaks_data, _, _ = jungfrau_frame_streaks_hist(
            streaks_per_module_dict,
            geom,
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
            msg=f"X] radius: {STREAKOGRAM_CONF_R_BIN_START}:{STREAKOGRAM_CONF_R_BIN_END} m ({STREAKOGRAM_CONF_R_BINS} bins)\n"
            + f"Y] intensity: {STREAKOGRAM_CONF_I_BIN_START}:{STREAKOGRAM_CONF_I_BIN_END} photons ({STREAKOGRAM_CONF_I_BINS} bins)",
            # aspect_ratio=STREAKOGRAM_CONF_I_BINS / STREAKOGRAM_CONF_R_BINS,
            aspect_ratio=1,
            history=1000,
            sum_over=True,
        )

        # powder histogram
        powder_hist2d, _, _ = jungfrau_powder_hist(
            streaks_per_module_dict,
            geom,
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
            msg=f"X] x_pos: {POWDER_CONF_X_BIN_START}:{POWDER_CONF_X_BIN_END} m ({POWDER_CONF_X_BINS} bins)\n"
            + f"Y] y_pos: {POWDER_CONF_Y_BIN_START}:{POWDER_CONF_Y_BIN_END} m ({POWDER_CONF_Y_BINS} bins)",
            history=1000,
            sum_over=True,
        )

    else:
        evt["analysis"]["isStreak"] = False

        # not_a_streak_hit = add_record(
        #     evt["analysis"],
        #     "analysis",
        #     "No Streak (raw)",
        #     image_data,
        # )
        masked_not_a_streak_hit = add_record(
            evt["analysis"],
            "analysis",
            "No Streak with mask",
            masked_img,
        )
        masked_not_a_streak_hit_cumulative = add_record(
            evt["analysis"],
            "analysis",
            "No Streak with mask (cumulative)",
            masked_img,
        )
        # plotting.image.plotImage(not_a_streak_hit, group="Non Hits", history=100)
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
