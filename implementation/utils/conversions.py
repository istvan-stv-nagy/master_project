from implementation.dataset.calib_data import CalibData
import numpy as np

from implementation.datastructures.pano_image import PanoImage


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

        x_image = np.arctan2(-y, x) / (horizontal_res * (np.pi / 180.0))
        y_image = -np.arctan2(z, d) / (vertical_res * (np.pi / 180.0))

        x_size = int(np.trunc((horizontal_fov[1] - horizontal_fov[0]) / horizontal_res))
        y_size = int(np.trunc((vertical_fov[1] - vertical_fov[0]) / vertical_res))

        x_offset = horizontal_fov[0] / horizontal_res
        x_image = np.trunc(x_image - x_offset).astype(np.int32)

        y_offset = vertical_fov[1] / vertical_res
        y_image = np.trunc(y_image + y_offset).astype(np.int32)

        mask = (x_image >= 0) & (x_image < x_size) & (y_image >= 0) & (y_image < y_size)
        x_image = x_image[mask]
        y_image = y_image[mask]

        x_road_image = np.zeros([y_size, x_size])
        y_road_image = np.zeros([y_size, x_size])
        z_road_image = np.zeros([y_size, x_size])

        road_coords = self.lidar_2_road(lidar_coords)
        y_road = road_coords[1, :]
        y_road = y_road[mask]
        z_road = road_coords[2, :]
        z_road = z_road[mask]
        x_road = road_coords[0, :]
        x_road = x_road[mask]

        x_road_image[y_image, x_image] = x_road
        y_road_image[y_image, x_image] = -y_road
        z_road_image[y_image, x_image] = z_road

        img_velo_x = np.zeros([y_size, x_size])
        img_velo_x[y_image, x_image] = x[mask]

        img_velo_y = np.zeros([y_size, x_size])
        img_velo_y[y_image, x_image] = y[mask]

        img_velo_z = np.zeros([y_size, x_size])
        img_velo_z[y_image, x_image] = z[mask]

        return PanoImage(
            x_road_img=x_road_image,
            y_road_img=y_road_image,
            z_road_img=z_road_image,
            x_velo_img=img_velo_x,
            y_velo_img=img_velo_y,
            z_velo_img=img_velo_z)

    def velo_2_image(self, point_cloud, width=1242, height=375):
        x, y, z = point_cloud
        points = np.vstack((x, y, z, np.ones(len(x))))
        c = self.calib.p2[:, :3]
        P = c @ (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
        camera_points = P @ (self.calib.tr_velo_to_cam @ points)
        camera_points[0] /= camera_points[-1]
        camera_points[1] /= camera_points[-1]
        camera_points[2] /= camera_points[-1]

        x_image = np.trunc(camera_points[0, :]).astype(np.int32)
        y_image = np.trunc(camera_points[1, :]).astype(np.int32)

        mask = (x_image >= 0) & (x_image < width) & (y_image >= 0) & (y_image < height)
        x_image = x_image[mask]
        y_image = y_image[mask]

        coords = np.vstack((x_image, y_image))
        return coords

