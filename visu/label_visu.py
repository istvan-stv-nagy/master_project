from datastructures.pano_image import PanoImage
from visu.plot_functions import *


class LabelVisu:
    def __init__(self):
        pass

    def show(self, pano_image:PanoImage, delimitations):
        plot_signals_limits([pano_image.get_channel(40), pano_image.get_channel(42)], [delimitations[40], delimitations[42]])
