import numpy as np


class Filter:
    def __init__(self):
        pass

    def filter(self, data):
        pass


class PointCloudFilter(Filter):
    def __init__(self, min_x=-np.inf, max_x=np.inf, min_y=-np.inf, max_y=np.inf, min_z=-np.inf, max_z=np.inf):
        super().__init__()
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.min_z = min_z
        self.max_z = max_z

    def filter(self, point_cloud):
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        r = point_cloud[:, 3]
        mask = (self.min_x < x) & (x < self.max_x) & \
               (self.min_y < y) & (y < self.max_y) & \
               (self.min_z < z) & (z < self.max_z)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        r = r[mask]
        return np.vstack((x, y, z, r)).T
