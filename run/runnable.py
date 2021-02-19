from convert.conversions import Converter
from data.data_reader import DataReader
from filter.filters import PointCloudFilter
from grid.grid_mapping import UniformGridGenerator, Grid
from processing.grid_processing import GridProcessingUnit
from visu.frame_visu import FrameVisu


class Runnable:
    def __init__(self):
        self.data_reader = DataReader()
        self.grid = Grid(rows=100, cols=80, row_res=0.5, col_res=0.5)
        self.grid_generator = UniformGridGenerator(self.grid)
        self.point_cloud_filter_lidar = PointCloudFilter(min_x=0)
        self.point_cloud_filter_road = PointCloudFilter(min_x=-20, max_x=20, min_z=0, min_y=-2, max_y=0.5)
        self.grid_processing_unit = GridProcessingUnit()

        self.visu = FrameVisu()

    def run(self, frame_count):
        frame_data = self.data_reader.read_frame(frame_count)

        converter = Converter(frame_data.calib)

        pts_lidar = self.point_cloud_filter_lidar.filter(frame_data.point_cloud)

        pano_image = converter.lidar_2_pano(pts_lidar, horizontal_fov=(-180, 180), vertical_fov=(-24.9, 5.0))

        pts_road = converter.lidar_2_road(frame_data.point_cloud)

        pts_road = self.point_cloud_filter_road.filter(pts_road)

        # depth_projection_image = converter.lidar_2_img(frame_data.point_cloud)

        height_grid = self.grid_generator.generate(pts_road)

        # processed_grid = self.grid_processing_unit.process_grid(height_grid)

        # elevation_image = converter.grid_2_image(self.grid, processed_grid)

        self.visu.show(frame_data, pts_road, height_grid, pano_image)
