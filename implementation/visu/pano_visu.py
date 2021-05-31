import os

import matplotlib.pyplot as plt
import numpy as np

from implementation.datastructures.pano_image import PanoImage


class PanoVisu:
    def __init__(self, save_fig=False, dump_path=r''):
        self.save_fig = save_fig
        self.dump_path = dump_path

    def show(self, index, pano: PanoImage):
        fig = plt.figure(figsize=(20, 3))
        min_pano_height = 0
        max_pano_height = 35
        pano_image = np.clip((pano.z_road_img - min_pano_height) / (max_pano_height - min_pano_height), 0.0, 1.0) * 255
        plt.imshow(pano_image, cmap='jet')
        if self.save_fig:
            fig.savefig(os.path.join(self.dump_path, "pano" + str(index) + ".png"))
