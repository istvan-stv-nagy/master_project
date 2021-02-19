from data.calib_data import CalibData
import numpy as np

from grid.grid_mapping import Grid


class Converter:
    def __init__(self, calib: CalibData):
        self.calib = calib
        self.lidar_2_cam2_mat = self.calib.p2 @ self.calib.r0_rect @ self.calib.tr_velo_to_cam

    def lidar_2_cam(self, lidar_coords):
        """
        Converts points from lidar to camera space (3D -> 3D)
        :param lidar_coords: points in lidar coordinate system
        :return: points in camera 3D coordinate system
        """
        r = np.copy(lidar_coords[3, :])
        lidar_coords[-1, :] = np.ones(len(lidar_coords[0]))
        cam_coords = self.calib.tr_velo_to_cam @ lidar_coords
        cam_coords[-1, :] = r
        return cam_coords

    def lidar_2_road(self, lidar_coords):
        """
        Converts points from lidar to road space (3D -> 3D)
        :param lidar_coords: points in lidar coordinate system
        :return: points in road 3D coordinate system
        """
        cam_coords = self.lidar_2_cam(lidar_coords)
        r = np.copy(cam_coords[3, :])
        cam_coords[-1, :] = np.ones(len(lidar_coords[0]))
        road_coords = self.calib.tr_cam_to_road @ cam_coords
        road_coords[-1, :] = r
        return road_coords

    def lidar_2_pano(self, lidar_coords, horizontal_fov, vertical_fov, horizontal_res=0.35, vertical_res=0.42, min_dist=0, max_dist=60):
        x = lidar_coords[0, :]
        y = lidar_coords[1, :]
        z = lidar_coords[2, :]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        x_image = np.arctan2(y, x) / (horizontal_res * (np.pi / 180))
        y_image = -np.arctan2(z, d) / (vertical_res * (np.pi / 180))

        x_size = int(np.ceil((horizontal_fov[1] - horizontal_fov[0]) / horizontal_res))
        y_size = int(np.ceil((vertical_fov[1] - vertical_fov[0]) / horizontal_res))

        x_offset = horizontal_fov[0] / horizontal_res
        x_image = np.trunc(x_image - x_offset).astype(np.int32)
        y_offset = vertical_fov[1] / vertical_res
        y_image = np.trunc(y_image + y_offset + 1).astype(np.int32)

        dist = (((max_dist - d) / (max_dist - min_dist)) * 255).astype(np.uint8)

        img = np.zeros([y_size + 1, x_size + 1], dtype=np.uint8)
        img[y_image, x_image] = x
        return img

    def lidar_2_img(self, point_cloud):
        x = point_cloud[0, :]
        y = point_cloud[1, :]
        z = point_cloud[2, :]
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
            row = int(ys[i])
            col = int(xs[i])
            pixel_image[row, col] = zs[i]
        return pixel_image

    def grid_2_image(self, grid:Grid, data):
        xs = []
        ys = []
        zs = []
        for row in range(grid.rows):
            for col in range(grid.cols):
                if not np.isnan(data[row, col]):
                    x = row * grid.row_res
                    y = col * grid.col_res - (grid.cols / 2 * grid.col_res)
                    z = data[row, col]
                    xs += [x]
                    ys += [y]
                    zs += [z]
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        point_cloud = np.vstack((xs, ys, zs))
        return self.lidar_2_img(point_cloud)
