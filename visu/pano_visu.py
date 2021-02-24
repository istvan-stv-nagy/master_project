from datastructures.pano_image import PanoImage
from visu.plot_functions import *


class PanoVisu:
    def __init__(self):
        self.min_pano_height = -0.2
        self.max_pano_height = 0.75

    def show(self, input_image, pano_image: PanoImage):
        pano = np.clip((pano_image.img - self.min_pano_height) / (self.max_pano_height - self.min_pano_height), 0.0, 1.0) * 255
        plot_images([input_image, pano])
        plot_signals([pano_image.get_channel(40), pano_image.get_channel(42), pano_image.get_channel(44), pano_image.get_channel(46)])