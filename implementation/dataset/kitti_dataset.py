import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from implementation.dataset.calib_data import CalibData


class FrameData:
    def __init__(self, image_color, gt_image, calib:CalibData, point_cloud):
        self.image_color = image_color
        self.gt_image = gt_image
        self.calib = calib
        self.point_cloud = point_cloud


class KittiDataset(Dataset):
    def __init__(self, velo_dir, image_dir, calib_dir, gt_dir, return_name=False):
        self.velo_dir = velo_dir
        self.image_dir = image_dir
        self.calib_dir = calib_dir
        self.gt_dir = gt_dir
        self.images = os.listdir(image_dir)
        self.return_name = return_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        gt_img_path = self.__get_gt_img_path(self.images[item])
        velo_path = os.path.join(self.velo_dir, self.images[item].replace(".png", ".bin"))
        calib_path = os.path.join(self.calib_dir, self.images[item].replace(".png", ".txt"))

        image_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt_image = None
        if gt_img_path is not None:
            gt_image = self.__read_gt_image(gt_img_path)

        calib: CalibData = self.__read_calib(calib_path)
        point_cloud = self.__read_velo(velo_path)
        frame_data = FrameData(
            image_color=image_color,
            gt_image=gt_image,
            calib=calib,
            point_cloud=point_cloud,
        )
        if not self.return_name:
            return frame_data
        else:
            return frame_data, self.images[item].replace(".png", "")

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

    def __get_gt_img_path(self, img_name):
        possibility_1 = img_name.replace("_", "_lane_")
        path1 = os.path.join(self.gt_dir, possibility_1)
        if os.path.exists(path1):
            return path1
        possibility_2 = img_name.replace("_", "_road_")
        path2 = os.path.join(self.gt_dir, possibility_2)
        if os.path.exists(path2):
            return path2
        return None

    @staticmethod
    def __read_gt_image(gt_img_path):
        gt_image = cv2.imread(gt_img_path)
        rgt = np.zeros((len(gt_image), len(gt_image[0])))
        class1 = np.array([255, 0, 255])
        for i in range(len(gt_image)):
            for j in range(len(gt_image[0])):
                pixel = np.array(gt_image[i][j])
                if np.array_equal(pixel, class1):
                    rgt[i][j] = 1
        return rgt
