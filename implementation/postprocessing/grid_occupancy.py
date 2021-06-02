import skimage.measure
import numpy as np


def get_grid_occupancy(coordinates, in_size, grid_cell_size=(2, 2)):
    original_grid = np.zeros(in_size)
    for coord in coordinates:
        original_grid[coord[1]][coord[0]] = 1

    result = skimage.measure.block_reduce(original_grid, grid_cell_size, np.sum) > 0
    result = result.repeat(grid_cell_size[0], axis=0).repeat(grid_cell_size[1], axis=1)
    return result

