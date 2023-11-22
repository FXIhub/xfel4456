import numpy as np


class PseudoPowderDiffraction:
    def __init__(self, geom):
        self.geom = geom
        self.xyz = geom.get_pixel_positions()
        self.blank_image = geom.output_array_for_position(dtype=int)
        self.blank_image[:] = -1
        _, self.center = geom.position_modules(
            np.zeros(geom.expected_data_shape, dtype=int), out=self.blank_image
        )
        self.reset()

    def get(self, positions):
        mod, ss, fs = positions.T
        x, y, z = self.xyz[mod, ss, fs].T / self.geom.pixel_size
        y0, x0 = self.center

        i = (x0 + x).astype(int)
        j = (y0 + y).astype(int)

        powder = self.blank_image.copy()
        powder[j, i] += 1

        self.powder_sum += powder
        self.count += 1

        return powder

    def get_sum(self):
        return self.powder_sum / self.count

    def reset(self):
        self.powder_sum = self.blank_image.copy()
        self.count = 0


class Peakogram:
    def __init__(self, geom, nr, rmin, rmax, nintens, intens_max):
        self.geom = geom
        self.xyz = geom.get_pixel_positions()
        self.nr = nr
        self.rmin = rmin
        self.rmax = rmax
        self.nintens = nintens
        self.intens_max = intens_max
        self.r_bins = np.linspace(rmin, rmax, nr + 1)
        self.i_bins = np.linspace(0, intens_max, nintens + 1)
        self.peakogram_sum = np.zeros([nintens, nr], float)

    def get(self, positions, intens):
        mod, ss, fs = positions.T
        x, y, z = self.xyz[mod, ss, fs].T / self.geom.pixel_size

        r = np.sqrt(x * x + y * y)
        peakogram, _, _ = np.histogram2d(intens, r, bins=(self.i_bins, self.r_bins))
        self.peakogram_sum += peakogram
        return peakogram

    def reset(self):
        self.peakogram_sum[:] = 0

    def get_sum(self):
        return self.peakogram_sum
