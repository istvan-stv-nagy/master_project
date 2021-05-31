from implementation.datastructures.pano_image import PanoImage
from visu.plot_functions import *


class PanoVisu:
    def __init__(self):
        self.min_pano_x = 0
        self.max_pano_x = 50

        self.min_pano_y = -5
        self.max_pano_y = 5

        self.min_pano_z = -1.9
        self.max_pano_z = -0.5

    def show(self, input_image, pano_image: PanoImage):
        pano_x = np.clip((pano_image.x_velo_img - self.min_pano_x) / (self.max_pano_x - self.min_pano_x), 0.0, 1.0) * 255
        pano_y = np.clip((pano_image.y_velo_img - self.min_pano_y) / (self.max_pano_y - self.min_pano_y), 0.0, 1.0) * 255
        pano_z = np.clip((pano_image.z_velo_img - self.min_pano_z) / (self.max_pano_z - self.min_pano_z), 0.0, 1.0) * 255
        plot_images([input_image, pano_x, pano_y, pano_z, pano_z])