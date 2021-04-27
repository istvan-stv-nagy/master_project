import os

import numpy as np

from labeling.labeling_params import LabelingParams
import matplotlib.pyplot as plt


def show_data(frame):
    pano = np.load(LabelingParams.OUTPUT_ROOT + "\\pano" + str(frame) + ".npy")
    min_pano_height = -0.2
    max_pano_height = 0.75
    pano_rgb = np.clip((pano - min_pano_height) / (max_pano_height - min_pano_height), 0.0, 1.0) * 255

    plt.imshow(pano_rgb, cmap='jet')

    if os.path.exists(LabelingParams.OUTPUT_ROOT + "\\pano_gt" + str(frame) + ".npy"):
        label = np.load(LabelingParams.OUTPUT_ROOT + "\\pano_gt" + str(frame) + ".npy")
        label[label == 0] = np.nan
        label[label == 1] = 0
        plt.imshow(label)
    plt.show()

if __name__ == '__main__':
    show_data(34)