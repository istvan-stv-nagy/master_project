import numpy as np


class CalibData:
    def __init__(self, raw_calib):
        self.p0 = np.reshape(raw_calib['P0'], (3, 4))
        self.p1 = np.reshape(raw_calib['P1'], (3, 4))
        self.p2 = np.reshape(raw_calib['P2'], (3, 4))
        self.p3 = np.reshape(raw_calib['P3'], (3, 4))

        r0 = np.zeros((4, 4))
        r0[3, 3] = 1
        r0[:3, :3] = np.reshape(raw_calib['R0_rect'], (3, 3))
        self.r0_rect = r0

        tr_velo_2_cam = np.zeros((4, 4))
        tr_velo_2_cam[3, 3] = 1
        tr_velo_2_cam[:3, :4] = np.reshape(raw_calib['Tr_velo_to_cam'], (3, 4))
        self.tr_velo_to_cam = tr_velo_2_cam

        tr_imu_2_velo = np.zeros((4, 4))
        tr_imu_2_velo[3, 3] = 1
        tr_imu_2_velo[:3, :4] = np.reshape(raw_calib['Tr_imu_to_velo'], (3, 4))
        self.tr_imu_to_velo = tr_imu_2_velo

        tr_cam_2_road = np.zeros((4, 4))
        tr_cam_2_road[3, 3] = 1
        tr_cam_2_road[:3, :4] = np.reshape(raw_calib['Tr_cam_to_road'], (3, 4))
        self.tr_cam_to_road = tr_cam_2_road
