from data.frame_data import FrameData
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mayavi.mlab as mlab


class FrameVisu:
    def __init__(self, frame_data: FrameData):
        self.frame_data = frame_data

    def show(self):
        fig, axs = plt.subplots(2)
        axs[0].imshow(self.frame_data.image_color)
        axs[1].imshow(self.frame_data.gt_image)
        plt.show()
        self.plot_point_cloud()
        print(self.frame_data.calib)

    def plot_point_cloud(self):
        point_cloud = self.frame_data.point_cloud
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        mlab.points3d(x, y, z, z, mode="point", colormap="spectral", figure=fig)
        mlab.show()
