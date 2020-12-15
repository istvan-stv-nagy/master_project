import numpy as np


class GridCell:
    def __init__(self):
        self.heights = np.array([])

    def add(self, z):
        self.heights = np.append(self.heights, [z])

    def mean_height(self):
        if self.is_valid():
            return np.mean(self.heights)
        return np.nan

    def median_height(self):
        if self.is_valid():
            return np.median(self.heights)
        return np.nan

    def count(self):
        return len(self.heights)

    def is_valid(self):
        return len(self.heights) > 0
