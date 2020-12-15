import cv2 as cv
import numpy as np


class GridProcessingUnit:
    def __init__(self):
        self.sobel_size = 3
        self.gradient_min_threshold = 0.05
        self.gradient_max_threshold = 0.5

    def process_grid(self, grid):
        gradient = self.gradient(grid)
        filtered_gradient = self.filtered_gradient(gradient)
        return filtered_gradient

    def gradient(self, grid):
        gradient = cv.Sobel(grid, cv.CV_64F, 1, 0, ksize=self.sobel_size)
        return np.abs(gradient)

    def filtered_gradient(self, grid_gradient):
        mask = (grid_gradient < self.gradient_min_threshold) | (grid_gradient > self.gradient_max_threshold)
        grid_gradient[mask] = np.nan
        return grid_gradient
