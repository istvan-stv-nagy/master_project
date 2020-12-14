import numpy as np


class CalibData:
    def __init__(self, raw_calib):
        self.p0 = np.reshape(raw_calib['P0'], (3, 4))
        self.p1 = np.reshape(raw_calib['P1'], (3, 4))
        self.p2 = np.reshape(raw_calib['P2'], (3, 4))
        self.p3 = np.reshape(raw_calib['P3'], (3, 4))
        self.r0_rect = np.reshape(raw_calib['R0_rect'], (3, 3))
        self.tr_velo_to_cam = np.reshape(raw_calib['Tr_velo_to_cam'], (3, 4))
        self.tr_imu_to_velo = np.reshape(raw_calib['Tr_imu_to_velo'], (3, 4))
        self.tr_cam_to_road = np.reshape(raw_calib['Tr_cam_to_road'], (3, 4))
