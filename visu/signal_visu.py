from implementation.datastructures.pano_image import PanoImage
from visu.plot_functions import plot_signals
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
import numpy as np


class SignalVisu:
    def __init__(self):
        pass

    def show(self, pano_image: PanoImage):
        sig = pano_image.get_channel(40)
        sig_gaussian = gaussian_filter1d(sig, sigma=1)
        sig_gradient = convolve(sig_gaussian, np.array([-1, 0, 1]))
        plot_signals([sig, sig_gaussian, sig_gradient])