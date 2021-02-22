from data.frame_data import FrameData
from visu.plot_functions import *


class InputVisu:
    def __init__(self):
        pass

    def show(self, frame_data: FrameData):
        plot_images([frame_data.image_color, frame_data.gt_image])
