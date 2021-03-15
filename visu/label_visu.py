from labeling.label_output import LabelOutput
from visu.plot_functions import *


class LabelVisu:
    def __init__(self):
        pass

    def show(self, label_output: LabelOutput):
        frames = [40, 52, 44, 46]
        channels = [label_output.data().get_channel(i) for i in frames]
        dels = [label_output.label()[i] for i in frames]
        plot_signals(channels + dels)
        plot_images([label_output.data().img, label_output.label()])
