import cv2 as cv
import os
import numpy as np

from data.calib_data import CalibData
from data.dataset_type import *
from data.frame_data import FrameData


class DataReader:

    def __init__(self, dataset_type=DatasetType.UNMARKED):
        self.dataset_type = dataset_type
        self.dataset_path = r'E:\Storage\7 Master Thesis\dataset'
        self.image_color_path = os.path.join(self.dataset_path, r'data_road\training\image_2')
        self.gt_image_path = os.path.join(self.dataset_path, r'data_road\training\gt_image_2')
        self.calib_path = os.path.join(self.dataset_path, r'data_road\training\calib')
        self.velodyne_path = os.path.join(self.dataset_path, r'velodyne\training\velodyne')

    def read_frame(self, frame_count):
        image_color = cv.imread(os.path.join(self.image_color_path, IMAGE_PREFIX[self.dataset_type] + self.get_frame_id(frame_count) + ".png"))
        gt_image = cv.imread(os.path.join(self.gt_image_path, GT_PREFIX[self.dataset_type] + self.get_frame_id(frame_count) + ".png"))
        point_cloud = self.read_point_cloud(os.path.join(self.velodyne_path, IMAGE_PREFIX[self.dataset_type] + self.get_frame_id(frame_count) + ".bin"))
        calib_raw = self.read_calib_file(os.path.join(self.calib_path, IMAGE_PREFIX[self.dataset_type] + self.get_frame_id(frame_count) + ".txt"))
        return FrameData(image_color, gt_image, point_cloud, CalibData(calib_raw))

    @staticmethod
    def get_frame_id(frame_count):
        return str(frame_count).zfill(6)

    def read_point_cloud(self, path):
        point_cloud = np.fromfile(str(path), dtype=np.float32, count=-1).reshape([-1, 4]).T
        return point_cloud

    def read_calib_file(self, path):
        data = {}
        with open(path, 'r') as file:
            for line in file.readlines():
                key, value = line.split(':', 1)
                data[key] = np.array([np.float(x) for x in value.split()])
        return data