from data.calib_data import CalibData
import numpy as np

from datastructures.pano_processing import PanoImage


class Converter:
    """
    Class handling conversions from one space to another
    Spaces include: lidar, camera, road, image
    """
    def __init__(self, calib: CalibData):
        self.calib = calib

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

    def lidar_2_pano(self, lidar_coords, horizontal_fov, vertical_fov, horizontal_res=0.35, vertical_res=0.42):
        """
        Converts lidar points in lidar space to a panoramic image specific to the
        lidar hardware used. Uses channel value and azimuth in order to place a 3D lidar
        point in the panoramic image
        :param lidar_coords: points in lidar coordinate system
        :param horizontal_fov: horizontal field of view (can be set by user)
        :param vertical_fov: vertical field of view (can be set by user)
        :param horizontal_res: horizontal resolution of sensor (hardware specific)
        :param vertical_res: vertical resolution of sensor (hardware specific)
        :return: panoramic height map corresponding to the input lidar points
        """
        x = lidar_coords[0, :]
        y = lidar_coords[1, :]
        z = lidar_coords[2, :]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        x_image = np.arctan2(-y, x) / (horizontal_res * (np.pi / 180))
        y_image = -np.arctan2(z, d) / (vertical_res * (np.pi / 180))

        x_size = int(np.ceil((horizontal_fov[1] - horizontal_fov[0]) / horizontal_res))
        y_size = int(np.ceil((vertical_fov[1] - vertical_fov[0]) / vertical_res))

        x_offset = horizontal_fov[0] / horizontal_res
        x_image = np.trunc(x_image - x_offset).astype(np.int32)
        y_offset = vertical_fov[1] / vertical_res
        y_image = np.trunc(y_image + y_offset + 1).astype(np.int32)

        mask = (x_image >= 0) & (x_image < x_size) & (y_image >= 0) & (y_image < y_size)
        x_image = x_image[mask]
        y_image = y_image[mask]

        img = np.zeros([y_size + 1, x_size + 1])
        img[:] = np.nan

        road_coords = self.lidar_2_road(lidar_coords)
        y_road = road_coords[1, :]
        y_road = y_road[mask]

        img[y_image, x_image] = -y_road
        return PanoImage(img)

    def lidar_2_img(self, point_cloud):
        x = point_cloud[0, :]
        y = point_cloud[1, :]
        z = point_cloud[2, :]
        point_coords = np.vstack((x, y, z, np.ones(len(x))))
        pixel_coords = self.calib.p2 @ self.calib.r0_rect @ self.calib.tr_velo_to_cam @ point_coords

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

    def lidar_2_image(self, point_cloud, width=1242, height=375):
        points_camera = self.lidar_2_cam(point_cloud)
        image_coords = self.calib.p2 @ self.calib.r0_rect @ points_camera
        image_pixels = image_coords / image_coords[2, :]
        image_pixels = image_pixels[:2, :]
        x_image = np.trunc(image_pixels[0, :]).astype(np.int32)
        y_image = np.trunc(image_pixels[1, :]).astype(np.int32)

        road_coords = self.lidar_2_road(point_cloud)
        y_road = road_coords[1, :]

        mask = (x_image >= 0) & (x_image < width) & (y_image >= 0) & (y_image < height)
        x_image = x_image[mask]
        y_image = y_image[mask]
        y_road = y_road[mask]

        image = np.zeros((height, width))
        image[:] = np.nan
        image[y_image, x_image] = y_road

        return image
