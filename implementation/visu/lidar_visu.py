import mayavi.mlab as mlab
import numpy as np


class LidarVisu:
    def __init__(self, x_range=(-np.inf, +np.inf), y_range=(-np.inf, +np.inf), z_range=(-np.inf, +np.inf)):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def show(self, point_cloud, values_to_plot):
        x = point_cloud[0, :]
        y = point_cloud[1, :]
        z = point_cloud[2, :]
        mask = (self.x_range[0] < x) & (x < self.x_range[1]) & \
               (self.y_range[0] < y) & (y < self.y_range[1]) & \
               (self.z_range[0] < z) & (z < self.z_range[1])
        x = x[mask]
        y = y[mask]
        z = z[mask]
        values_to_plot = values_to_plot[mask]
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        mlab.points3d(x, y, z, values_to_plot, mode="point", colormap="spectral", figure=fig)
