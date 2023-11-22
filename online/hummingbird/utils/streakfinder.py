import os
import sys
import numpy as np


prop_dir = "/gpfs/exfel/exp/SPB/202302/p004456"
streakfinder_path = os.path.join(prop_dir, "usr/Shared/gennaroi/xfel4456/offline")
sys.path.append(streakfinder_path)
from hit_finding_utils import single_streak_finder


class StreakFinder:
    def __init__(self, thr_percent, min_pixels, max_area):
        self.thr_percent = thr_percent
        self.min_pixels = min_pixels
        self.max_area = max_area

    def find_streaks(self, image, mask=None):

        masked_data = image.copy()
        masked_data[~mask] = 0.0

        threshold = np.nanpercentile(masked_data, self.thr_percent)

        num_streaks = 0
        positions = []
        intens = []

        nmod = len(image)
        for modno in range(nmod):
            labels, centroids, props, _, _ = single_streak_finder(
                img_array=masked_data[modno],
                thld=threshold,
                min_pix=self.min_pixels,
                area_max_size=self.max_area,
            )
            count = len(centroids)
            num_streaks += count

            props = [props[lab - 1] for lab in labels]
            intens_max = np.array([p.intensity_max for p in props])
            intens_mean = np.array([p.intensity_mean * p.area for p in props])

            intens.append(np.hstack([intens_max[:, None], intens_mean[:, None]]))

            positions.append(
                np.hstack(
                    [np.full([count, 1], modno, dtype=int), centroids.astype(int)]
                )
            )

        positions = np.vstack(positions)
        intens = np.vstack(intens)

        return type(
            "Streaks",
            (),
            dict(num_streaks=num_streaks, positions=positions, intens=intens),
        )
