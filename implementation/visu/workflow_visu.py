import matplotlib.pyplot as plt
import numpy as np
from implementation.datastructures.pano_image import PanoImage
from implementation.visu.plot_utils import PlotUtils
import os


class WorkFlowVisu:
    def __init__(self, save_fig=False, dump_path=r''):
        self.save_fig = save_fig
        self.dump_path = dump_path

    def show(self, index, image_color, pano: PanoImage, prediction, projection_coordinates):
        fig = plt.figure(constrained_layout=True, figsize=(22, 8))
        gs = fig.add_gridspec(2, 2)
        pano_visu = fig.add_subplot(gs[0, 0])
        prediction_visu = fig.add_subplot(gs[0, 1])
        output_visu = fig.add_subplot(gs[1, :])

        min_pano_height = -0.2
        max_pano_height = 0.75
        pano_image = np.clip((pano.y_road_img - min_pano_height) / (max_pano_height - min_pano_height), 0.0, 1.0) * 255
        pano_visu.imshow(pano_image, cmap='jet')

        prediction_visu.imshow(prediction)

        output_image = PlotUtils.projection_image(image_color, projection_coordinates)
        output_visu.imshow(output_image)

        if self.save_fig:
            fig.savefig(os.path.join(self.dump_path, "result" + str(index) + ".png"))

