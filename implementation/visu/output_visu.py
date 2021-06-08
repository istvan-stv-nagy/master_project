import os

import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from implementation.datastructures.freespace_output import FreespaceOutput


class OutputVisu:
    def __init__(self, save_fig=False, dump_path=r''):
        self.save_fig = save_fig
        self.dump_path = dump_path

    def show(self, index, image, freespace: FreespaceOutput, fill=True):
        fig = plt.figure(figsize=(10, 3))
        plt.axis('off')
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        plt.imshow(rgb_image)
        if fill:
            occupancy_image = np.ma.masked_array(freespace.mask, freespace.mask == 0)
            norm = colors.Normalize(vmin=0, vmax=2)
            plt.imshow(occupancy_image, cmap='Greens', alpha=0.7, norm=norm)
        else:
            poly_xs, poly_ys = freespace.poly.exterior.xy
            plt.plot(poly_xs, poly_ys, color='yellowgreen', linewidth=3)
        if self.save_fig:
            fig.savefig(os.path.join(self.dump_path, "output" + str(index) + ".png"))

