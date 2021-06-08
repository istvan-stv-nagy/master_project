from shapely.geometry import Polygon, Point
import numpy as np


class FreespaceOutput:
    def __init__(self, left_points, right_points, top_points, mask_size):
        self.left_points = sorted(left_points, reverse=True)
        self.right_points = sorted(right_points, reverse=False)
        self.top_points = top_points
        points = self.left_points + self.top_points + self.right_points
        points = [x[::-1] for x in points]
        self.poly: Polygon = Polygon(points)
        self.mask = np.zeros(mask_size)
        for i in range(mask_size[0]):
            for j in range(mask_size[1]):
                if self.poly.contains(Point(j, i)):
                    self.mask[i][j] = 1
