import matplotlib.pyplot as plt
import numpy as np
from implementation.datastructures.pano_image import PanoImage
from implementation.visu.plot_utils import PlotUtils
import os
import cv2


class WorkFlowVisu:
    def __init__(self, save_fig=False, dump_path=r''):
        self.save_fig = save_fig
        self.dump_path = dump_path

    def show(self, index, image_color, pano: PanoImage, prediction, projection_coordinates, occupancy):
        fig = plt.figure(constrained_layout=True, figsize=(22, 8))
        gs = fig.add_gridspec(2, 2)
        pano_visu = fig.add_subplot(gs[0, 0])
        prediction_visu = fig.add_subplot(gs[0, 1])
        projection_visu = fig.add_subplot(gs[1, 0])
        occupancy_visu = fig.add_subplot(gs[1, 1])

        min_pano_height = -0.2
        max_pano_height = 0.75
        pano_image = np.clip((pano.y_road_img - min_pano_height) / (max_pano_height - min_pano_height), 0.0, 1.0) * 255
        pano_visu.imshow(pano_image, cmap='jet')

        prediction_visu.imshow(prediction)

        projection_image = PlotUtils.projection_image(image_color, projection_coordinates)
        projection_visu.imshow(projection_image)

        occupancy_visu.imshow(image_color)
        occupancy_image = np.ma.masked_array(occupancy, occupancy == 0)
        occupancy_visu.imshow(occupancy_image, alpha=0.7)

        if self.save_fig:
            fig.savefig(os.path.join(self.dump_path, "result" + str(index) + ".png"))

