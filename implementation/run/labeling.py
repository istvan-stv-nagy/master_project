import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point

from implementation.dataset.kitti_dataset import KittiDataset, FrameData
from implementation.datastructures.pano_image import PanoImage
from implementation.utils.conversions import Converter

TRAINING_IMAGE_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\image_2'
TRAINING_CALIB_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\calib'
TRAINING_VELO_DIR = r'E:\Storage\7 Master Thesis\dataset\velodyne\training\velodyne'
TRAINING_GT_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\gt_image_2'


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
        pano = np.clip((pano_image.y_road_img - self.min_pano_height) / (self.max_pano_height - self.min_pano_height), 0.0, 1.0) * 255
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


def main():
    dataset = KittiDataset(
        image_dir=TRAINING_IMAGE_DIR,
        calib_dir=TRAINING_CALIB_DIR,
        velo_dir=TRAINING_VELO_DIR,
        gt_dir=TRAINING_GT_DIR
    )

    for i in range(98, dataset.__len__()):
        frame_data: FrameData = dataset[i]
        converter: Converter = Converter(calib=frame_data.calib)
        pano: PanoImage = converter.lidar_2_pano(
            lidar_coords=frame_data.point_cloud,
            horizontal_fov=(-90, 90),
            vertical_fov=(-24.9, 2.0)
        )
        labeling_tool = SemanticLabeling()
        label = labeling_tool.label(pano)
        pano_image = Image.fromarray(pano.y_road_img)
        pano_image.save(r'E:\Storage\7 Master Thesis\dataset\semseg\train\image' + "\\uut" + str(i) + ".tif")
        label_image = Image.fromarray(label)
        label_image.save(r'E:\Storage\7 Master Thesis\dataset\semseg\train\mask' + "\\uut" + str(i) + ".tif")


if __name__ == '__main__':
    main()
