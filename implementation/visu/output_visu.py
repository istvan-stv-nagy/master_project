import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import numpy as np


class OutputVisu:
    def __init__(self):
        pass

    def show(self, image, occupancy):
        plt.figure()
        plt.axis('off')
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        plt.imshow(rgb_image)
        occupancy_image = np.ma.masked_array(occupancy, occupancy == 0)
        norm = colors.Normalize(vmin=0, vmax=2)
        plt.imshow(occupancy_image, cmap='Greens', alpha=0.7, norm=norm)

