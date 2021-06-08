from typing import List

import skimage.measure
import numpy as np
from shapely.geometry import Polygon, Point

from implementation.datastructures.freespace_output import FreespaceOutput


class GridCell:
    def __init__(self):
        self.points = []

    def count(self):
        return len(self.points)

    def add(self, point):
        self.points.append(point)

    def upmost_coordinate(self):
        return sorted(self.points)[0]

    def downmost_coordinate(self):
        return sorted(self.points)[-1]

    def leftmost_coordinate(self):
        return sorted(self.points, key=lambda x: x[1])[0]

    def rightmost_coordinate(self):
        return sorted(self.points, key=lambda x: x[1])[-1]


def get_grid_occupancy(coordinates, in_size, grid_cell_size=(2, 2)):
    grid: List[List[GridCell]] = []
    grid_size = (int(np.ceil(in_size[0] / grid_cell_size[0])), int(np.ceil(in_size[1] / grid_cell_size[1])))
    for i in range(grid_size[0]):
        grid += [[]]
        for j in range(grid_size[1]):
            grid[i] += [GridCell()]

    original_grid = np.zeros(in_size)
    for coord in coordinates:
        original_grid[coord[1]][coord[0]] = 1
        grid_row = coord[1] // grid_cell_size[0]
        grid_col = coord[0] // grid_cell_size[1]
        grid[grid_row][grid_col].add((coord[1], coord[0]))

    coarse_occupancy = skimage.measure.block_reduce(original_grid, grid_cell_size, np.sum)
    coarse_occupancy = np.clip(0, coarse_occupancy, 1)
    coarse_occupancy_lowres = coarse_occupancy.copy()

    left_points = []
    right_points = []
    top_points = []

    for i in range(len(coarse_occupancy_lowres)):
        first = None
        second = None
        for j in range(0, len(coarse_occupancy_lowres[i])):
            if coarse_occupancy_lowres[i][j] == 1:
                first = (i, j)
                break
        for j in range(len(coarse_occupancy_lowres[i]) - 1, -1, -1):
            if coarse_occupancy_lowres[i][j] == 1:
                second = (i, j)
                break
        if (first is not None) and (second is not None) and (first != second):
            left_points += [grid[first[0]][first[1]].leftmost_coordinate()]
            right_points += [grid[second[0]][second[1]].rightmost_coordinate()]

    left_end = max([p[1] for p in left_points]) // in_size[1]
    right_start = min([p[1] for p in right_points]) // in_size[1]

    for j in range(left_end + 1, right_start):
        top = None
        for i in range(0, len(coarse_occupancy_lowres)):
            if coarse_occupancy_lowres[i][j] == 1:
                top = (i, j)
                break
        if top is not None:
            top_points += [grid[top[0]][top[1]].upmost_coordinate()]

    return FreespaceOutput(left_points, right_points, top_points, in_size)





