import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from implementation.dataset.calib_data import CalibData


class FrameData:
    def __init__(self, image_color, calib:CalibData, point_cloud):
        self.image_color = image_color
        self.calib = calib
        self.point_cloud = point_cloud


class KittiDataset(Dataset):
    def __init__(self, velo_dir, image_dir, calib_dir):
        self.velo_dir = velo_dir
        self.image_dir = image_dir
        self.calib_dir = calib_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        velo_path = os.path.join(self.velo_dir, self.images[item].replace(".png", ".bin"))
        calib_path = os.path.join(self.calib_dir, self.images[item].replace(".png", ".txt"))

        image_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        calib: CalibData = self.__read_calib(calib_path)
        point_cloud = self.__read_velo(velo_path)
        return FrameData(
            image_color=image_color,
            calib=calib,
            point_cloud=point_cloud
        )

    @staticmethod
    def __read_calib(path):
        data = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                key, value = line.split(':', 1)
                data[key] = np.array([np.float(x) for x in value.split()])
        return CalibData(raw_calib=data)

    @staticmethod
    def __read_velo(path):
        point_cloud = np.fromfile(str(path), dtype=np.float32, count=-1).reshape([-1, 4]).T
        return point_cloud
