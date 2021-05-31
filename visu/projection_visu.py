from visu.plot_functions import *
import cv2


class ProjectionVisu:
    def __init__(self):
        pass

    def show(self, gt_image, projection_image):
        plt.imshow(gt_image, alpha=0.9)
        plt.imshow(projection_image, alpha=0.75)

    def print_projection_plt(self, points, color, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i in range(points.shape[1]):
            cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
