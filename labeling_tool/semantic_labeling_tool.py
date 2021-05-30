from convert.conversions import Converter
from data.data_reader import DataReader
from datastructures.pano_image import PanoImage
from filter.filters import PointCloudFilter
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from PIL import Image


class SemanticLabeling:
    def __init__(self):
        self.min_pano_height = -0.2
        self.max_pano_height = 0.75
        self.pts = []
        self.fig, self.ax = plt.subplots(figsize=(18, 10))
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.poly = None

    def onclick(self, event):
        if event.button == 1:
            self.pts += [(int(event.xdata), int(event.ydata))]
        elif event.button == 3:
            self.poly = Polygon(self.pts)
            poly_xs, poly_ys = self.poly.exterior.xy
            plt.gca().plot(poly_xs, poly_ys)
            plt.draw()
        elif event.button == 2:
            print("exit")
            plt.gcf().canvas.mpl_disconnect(self.cid)
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        plt.gca().plot(xs, ys, 'k-')
        plt.draw()

    def label(self, pano_image: PanoImage):
        pano = np.clip((pano_image.y_img - self.min_pano_height) / (self.max_pano_height - self.min_pano_height), 0.0, 1.0) * 255
        self.ax.imshow(pano, cmap='jet')
        plt.show()
        return self.to_label_image()

    def to_label_image(self):
        label = np.zeros((64, 514))
        print(self.poly.exterior)
        for i in range(64):
            for j in range(514):
                if self.poly.contains(Point(j, i)):
                    label[i][j] = 1
        return label


class Runnable:
    def __init__(self):
        self.data_reader = DataReader()
        self.point_cloud_filter_road = PointCloudFilter(min_x=-20, max_x=20, min_z=0, min_y=-2, max_y=0.5)

    def get_pano(self, frame_count):
        frame_data = self.data_reader.read_frame(frame_count)

        converter = Converter(frame_data.calib)

        pano_image = converter.lidar_2_pano(
            frame_data.point_cloud,
            horizontal_fov=(-90, 90), vertical_fov=(-24.9, 2.0))

        return pano_image


if __name__ == '__main__':
    for i in range(84, 100):
        r = Runnable()
        pan = r.get_pano(i)
        lab = SemanticLabeling()
        l = lab.label(pan)
        # fig, axs = plt.subplots(2)
        # axs[0].imshow(pan.y_img)
        # axs[1].imshow(l)
        # plt.show()
        pano_image = Image.fromarray(pan.y_img)
        pano_image.save(r'E:\Storage\7 Master Thesis\dataset\semseg\train\image' + "\\uut" + str(i) + ".tif")
        label_image = Image.fromarray(l)
        label_image.save(r'E:\Storage\7 Master Thesis\dataset\semseg\train\mask' + "\\uut" + str(i) + ".tif")

