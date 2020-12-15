import numpy as np

from grid.grid_cell import GridCell


class GridGenerator:
    def __init__(self, rows=50, cols=50):
        self.rows = rows
        self.cols = cols
        mat = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row += [GridCell()]
            mat += [row]
        self.grid = np.array(mat, dtype=object)

    def generate(self, point_cloud):
        pass

    def point_2_cell(self, x, y, z):
        pass


class UniformGridGenerator(GridGenerator):
    def __init__(self, rows, cols, rows_res, cols_res):
        super().__init__(rows, cols)
        self.row_res = rows_res
        self.cols_res = cols_res

    def generate(self, point_cloud):
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        for i in range(len(x)):
            self.__point_2_cell(x[i], y[i], z[i])
        return self.__height_map()

    def __point_2_cell(self, x, y, z):
        row = int(x / self.row_res)
        col = int(-y / self.cols_res + (self.cols / 2))
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row, col].add(z)

    def __height_map(self):
        grid_flat = self.grid.flatten()
        heights = np.array([cell.median_height() for cell in grid_flat])
        height_map = heights.reshape(self.grid.shape)
        return height_map
