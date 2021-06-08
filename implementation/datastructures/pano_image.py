import numpy as np


class PanoImage:
    def __init__(self, x_road_img, y_road_img, z_road_img, x_velo_img, y_velo_img, z_velo_img):
        self.x_road_img = x_road_img
        self.y_road_img = y_road_img
        self.z_road_img = z_road_img
        self.x_velo_img = x_velo_img
        self.y_velo_img = y_velo_img
        self.z_velo_img = z_velo_img
        self.lidar_dist = np.sqrt(np.square(x_velo_img) + np.square(y_velo_img) + np.square(z_velo_img))

    def velo(self, mask=None):
        return self.x_velo_img[mask].flatten(), self.y_velo_img[mask].flatten(), self.z_velo_img[mask].flatten()
