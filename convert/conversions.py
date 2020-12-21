from data.calib_data import CalibData
import numpy as np


class Converter:
    def __init__(self, calib: CalibData):
        self.calib = calib
        self.lidar_2_cam2_mat = self.calib.p2 @ self.calib.r0_rect @ self.calib.tr_velo_to_cam

    def lidar_2_cam(self, x, y, z):
        point_coords = np.vstack((x, y, z, np.ones(len(x))))
        pixel_coords = self.lidar_2_cam2_mat @ point_coords
        pixel_coords_normalized = pixel_coords / pixel_coords[2:]
        xs = pixel_coords_normalized[0, :]
        ys = pixel_coords_normalized[1, :]
        mask = (xs >= 0) & (xs < 1242) & (ys >= 0) & (ys < 375)
        xs = xs[mask]
        ys = ys[mask]
        zs = x[mask]
        pixel_image = np.zeros((375, 1242))
        for i in range(len(xs)):
            pixel_image[int(ys[i]), int(xs[i])] = zs[i]
        return pixel_image


