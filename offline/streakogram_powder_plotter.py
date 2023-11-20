import argparse
import pathlib
from typing import Tuple

import extra_geom
import h5py
import matplotlib.pyplot as plt
import numpy as np

FRAME_SHAPE = (8, 512, 1024)
streak_dtype = np.dtype(
    [
        ("frame_index", np.uint32),
        ("pixel_coord", np.uint32),
        ("intensity", np.float32),
        ("area", np.uint16),
    ]
)

parser = argparse.ArgumentParser(
    description="Generate the powder and streakogram plots using a events h5 file generated with `streak_finder.py`."
    + " Requires paths to events file and geometry file."
)
parser.add_argument(
    "-e",
    "--events_file",
    help="events.h5 file generated with `streak_finder.py`",
    type=str,
    required=True,
)
parser.add_argument(
    "-g", "--geom_file", help="Path to the Jungfrau geom file", type=str, required=True
)

parser.add_argument(
    "-m", "--beam_center_x", help="Beam center X in m", type=float, default=0
)
parser.add_argument(
    "-n", "--beam_center_y", help="Beam center Y in m", type=float, default=0
)
parser.add_argument(
    "-y", "--ybins", help="Y bins (for powder plot)", type=int, default=100
)
parser.add_argument(
    "-x", "--xbins", help="X bins (for powder plot)", type=int, default=100
)
parser.add_argument(
    "-r", "--rbins", help="R bins (for streakogram plot)", type=int, default=100
)
parser.add_argument(
    "-i", "--ibins", help="I bins (for streakogram plot)", type=int, default=100
)
args = parser.parse_args()

# check mandatory args.events_file
if not pathlib.Path(args.events_file).exists():
    raise Exception(f"{args.events_file=} does not exist")

# check mandatory args.geom_file
if not pathlib.Path(args.geom_file).exists():
    raise Exception(f"{args.geom_file=} does not exist")

args.output_file_powder = pathlib.Path(args.events_file).with_suffix(
    ".streaks_powder_plot.png"
)
args.output_file_streakograms = pathlib.Path(args.events_file).with_suffix(
    ".streakogram_plot.png"
)


def jungfrau_pixel_physical_displacements(
    streak: streak_dtype,
    jungfrau_pixel_physical_positions: np.ndarray,
    beam_center_physical: dict[str, float],
) -> dict[str, float]:
    module_index = streak["pixel_coord"] // (FRAME_SHAPE[1] * FRAME_SHAPE[2])
    pixel_x_in_module = streak["pixel_coord"] // (
        FRAME_SHAPE[1] * FRAME_SHAPE[2] % FRAME_SHAPE[2]
    )
    pixel_y_in_module = streak["pixel_coord"] // (
        FRAME_SHAPE[1] * FRAME_SHAPE[2] % FRAME_SHAPE[1]
    )
    pixel_positions_this_module = jungfrau_pixel_physical_positions[module_index]
    return {
        "x": pixel_positions_this_module[pixel_y_in_module, pixel_x_in_module][0]
        - beam_center_physical["x"],
        "y": pixel_positions_this_module[pixel_y_in_module, pixel_x_in_module][1]
        - beam_center_physical["y"],
    }


def get_streakogram_config_dict(
    ints: list[float],
    rs: list[float],
    r_bins: int = 100,
    i_bins: int = 100,
) -> dict[str, np.ndarray]:
    # streakogram parameters ---------------------
    r_bin_start = min(rs)
    r_bin_end = max(rs)

    i_bin_start = min(ints)
    i_bin_end = max(ints)

    return {
        "r_bins": np.linspace(
            r_bin_start,
            r_bin_end,
            r_bins,
        ),
        "i_bins": np.linspace(
            i_bin_start,
            i_bin_end,
            i_bins,
        ),
    }


def get_powder_plot_config_dict(
    x_displacements: list[float],
    y_displacements: list[float],
    x_bins: int = 100,
    y_bins: int = 100,
) -> dict[str, np.ndarray]:
    x_bin_start = min(x_displacements)
    x_bin_end = max(x_displacements)

    y_bin_start = min(y_displacements)
    y_bin_ebd = max(y_displacements)

    return {
        "x_bins": np.linspace(
            x_bin_start,
            x_bin_end,
            x_bins,
        ),
        "y_bins": np.linspace(
            y_bin_start,
            y_bin_ebd,
            y_bins,
        ),
    }


def compute_streakogram_plot_histogram(
    ints: list[float],
    rs: list[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    streakogram_plot_conf = get_streakogram_config_dict(
        ints=ints,
        rs=rs,
        r_bins=args.rbins,
        i_bins=args.ibins,
    )
    h, x_bins_edges, y_bins_edges = np.histogram2d(
        rs,
        ints,
        bins=(streakogram_plot_conf["r_bins"], streakogram_plot_conf["i_bins"]),
    )
    h = h.T
    x, y = np.meshgrid(x_bins_edges, y_bins_edges)
    return h, x, y


def compute_powder_plot_histogram(
    x_displacements: list[float],
    y_displacements: list[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    powder_plot_conf = get_powder_plot_config_dict(
        x_displacements=x_displacements,
        y_displacements=y_displacements,
        x_bins=args.xbins,
        y_bins=args.ybins,
    )
    h, x_bins_edges, y_bins_edges = np.histogram2d(
        x_displacements,
        y_displacements,
        bins=(powder_plot_conf["x_bins"], powder_plot_conf["y_bins"]),
    )
    h = h.T
    x, y = np.meshgrid(x_bins_edges, y_bins_edges)
    return h, x, y


jungfrau_geom = extra_geom.JUNGFRAUGeometry.from_crystfel_geom(args.geom_file)
beam_center_m = {"x": args.beam_center_x, "y": args.beam_center_y}

with h5py.File(args.events_file, "r") as h5:
    streak_list = np.asarray(h5["streak_list"], dtype=streak_dtype)

print(f"found {len(streak_list)} streaks in {args.events_file}")

# compute streaks centroids physical positions in x and y [m]
jps = jungfrau_geom.get_pixel_positions()
pixel_displacements = [
    jungfrau_pixel_physical_displacements(
        streak=s,
        jungfrau_pixel_physical_positions=jps,
        beam_center_physical=beam_center_m,
    )
    for s in streak_list
]
x_displacements, y_displacements = np.array(
    [p["x"] for p in pixel_displacements]
), np.array([p["y"] for p in pixel_displacements])
rs = np.sqrt(
    (x_displacements - beam_center_m["x"]) ** 2
    + (y_displacements - beam_center_m["y"]) ** 2
)
ints = np.array([s["intensity"] for s in streak_list])

# powder plot
hist_powder, x_axis, y_axis = compute_powder_plot_histogram(
    x_displacements, y_displacements
)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax_colormesh = ax.pcolormesh(
    x_axis,
    y_axis,
    hist_powder,
    cmap="viridis",
    shading="auto",
)
plt.colorbar(ax_colormesh)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
plt.savefig(args.output_file_powder, bbox_inches="tight")
print(f"powder plot saved to {args.output_file_powder}")


# streakogram plot
hist_streakogram, i_axis, r_axis = compute_streakogram_plot_histogram(ints, rs)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax_colormesh = ax.pcolormesh(
    r_axis,
    i_axis,
    hist_streakogram,
    cmap="viridis",
    shading="auto",
)
plt.colorbar(ax_colormesh)
ax.set_xlabel("r [m]")
ax.set_ylabel("I [photons]")
plt.savefig(args.output_file_streakograms, bbox_inches="tight")
print(f"streakogram plot saved to {args.output_file_streakograms}")
