import numpy as np

from grid.grid_cell import GridCell


class Grid:
    def __init__(self, rows=50, cols=50, row_res=0.5, col_res=0.5):
        self.rows = rows
        self.cols = cols
        self.row_res = row_res
        self.col_res = col_res


class GridGenerator:
    def __init__(self, grid: Grid):
        self.grid = grid
        mat = []
        for i in range(grid.rows):
            row = []
            for j in range(grid.cols):
                row += [GridCell()]
            mat += [row]
        self.map = np.array(mat, dtype=object)

    def generate(self, point_cloud):
        pass

    def __point_2_cell(self, x, y, z):
        pass


class UniformGridGenerator(GridGenerator):
    def __init__(self, grid: Grid):
        super().__init__(grid)

    def generate(self, point_cloud):
        x = point_cloud[0, :]
        y = point_cloud[1, :]
        z = point_cloud[2, :]
        for i in range(len(x)):
            self.__point_2_cell(x[i], y[i], z[i])
        return self.__height_map()

    def __point_2_cell(self, x, y, z):
        row = int(z / self.grid.row_res)
        col = int(x / self.grid.col_res + (self.grid.cols / 2))
        if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
            self.map[row, col].add(y)

    def __height_map(self):
        grid_flat = self.map.flatten()
        heights = np.array([cell.median_height() for cell in grid_flat])
        height_map = heights.reshape(self.map.shape)
        return height_map
