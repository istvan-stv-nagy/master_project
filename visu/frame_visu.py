import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np

from data.frame_data import FrameData


class FrameVisu:
    def __init__(self):
        pass

    def show(self, frame_data: FrameData, filtered_point_cloud, height_grid, processed_grid, depth_projection_image):
        self.__plot_input_images(frame_data.image_color, frame_data.gt_image)

        self.__plot_point_cloud(filtered_point_cloud)

        self.__plot_grids(height_grid, processed_grid)

        self.__plot_point_cloud_projection(frame_data.image_color, depth_projection_image)

        plt.show()
        mlab.show()

    @staticmethod
    def __plot_input_images(image_color, gt_image):
        fig, axs = plt.subplots(2)
        axs[0].imshow(image_color)
        axs[1].imshow(gt_image)

    @staticmethod
    def __plot_point_cloud(point_cloud):
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        mlab.points3d(x, y, z, z, mode="point", colormap="spectral", figure=fig)

    @staticmethod
    def __plot_grids(height_grid, processed_grid):
        fig, axs = plt.subplots(2)
        axs[0].set_ylim(0, len(height_grid))
        axs[0].imshow(height_grid)
        axs[1].set_ylim(0, len(height_grid))
        axs[1].imshow(processed_grid)

    @staticmethod
    def __plot_point_cloud_projection(image, depth_image):
        plt.figure()
        plt.imshow(image, alpha=1.0)
        depth_image = np.ma.masked_where(depth_image == 0, depth_image)
        plt.imshow(depth_image, alpha=0.9)
