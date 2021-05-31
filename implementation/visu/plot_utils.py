import cv2
import numpy as np


class PlotUtils:
    @staticmethod
    def projection_image(original_image, points):
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (255, 255, 255), -1)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)