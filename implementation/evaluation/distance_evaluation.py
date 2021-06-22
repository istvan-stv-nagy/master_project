from implementation.datastructures.pano_image import PanoImage
import numpy as np


class DistanceEvaluation:
    def __init__(self):
        pass

    def run(self, prediction, ground_truth, pano: PanoImage):
        tp_locations = np.where((prediction == 1) & (ground_truth == 1))
        x_img = pano.x_velo_img
        x_valid = x_img[tp_locations]
        print(x_valid.flatten())