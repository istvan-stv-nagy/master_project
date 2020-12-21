from convert.conversions import Converter
from data.data_reader import DataReader
from filter.filters import PointCloudFilter
from grid.grid_mapping import UniformGridGenerator
from processing.grid_processing import GridProcessingUnit
from visu.frame_visu import FrameVisu


class Runnable:
    def __init__(self):
        self.data_reader = DataReader()
        self.grid_generator = UniformGridGenerator(rows=100, cols=80, rows_res=0.5, cols_res=0.5)
        self.point_cloud_filter = PointCloudFilter(min_x=0, max_z=-0.5, min_z=-3, min_y=-20, max_y=20)
        self.grid_processing_unit = GridProcessingUnit()

        self.visu = FrameVisu()

    def run(self, frame_count):
        frame_data = self.data_reader.read_frame(frame_count)

        converter = Converter(frame_data.calib)

        point_cloud = self.point_cloud_filter.filter(frame_data.point_cloud)

        depth_projection_image = converter.lidar_2_cam(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

        height_grid = self.grid_generator.generate(point_cloud)

        processed_grid = self.grid_processing_unit.process_grid(height_grid)

        self.visu.show(frame_data, point_cloud, height_grid, processed_grid, depth_projection_image)
