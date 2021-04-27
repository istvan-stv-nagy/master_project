from convert.conversions import Converter
from data.data_reader import DataReader
from filter.filters import PointCloudFilter
from labeling.label_pano import LabelPanoTool
from labeling.labeling_params import LabelingParams
from visu.input_visu import InputVisu
from visu.label_visu import LabelVisu
from visu.pano_visu import PanoVisu
from visu.lidar_visu import LidarVisu
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
from visu.projection_visu import ProjectionVisu
from visu.signal_visu import SignalVisu


class Runnable:
    def __init__(self):
        self.data_reader = DataReader()
        self.point_cloud_filter_road = PointCloudFilter(min_x=-20, max_x=20, min_z=0, min_y=-2, max_y=0.5)

        self.input_visu = InputVisu()
        self.pano_visu = PanoVisu()
        self.lidar_visu = LidarVisu()
        self.projection_visu = ProjectionVisu()
        self.signal_visu = SignalVisu()

        self.label_pano_tool = LabelPanoTool()
        self.label_visu = LabelVisu()

    def run(self, frame_count):
        frame_data = self.data_reader.read_frame(frame_count)

        converter = Converter(frame_data.calib)

        pano_image = converter.lidar_2_pano(
            frame_data.point_cloud,
            horizontal_fov=(-90, 90), vertical_fov=(-24.9, 2.0))

        #np.save(LabelingParams.OUTPUT_ROOT + "\\pano" + str(frame_count) + ".npy", pano_image.img)

        #pts_road = converter.lidar_2_road(frame_data.point_cloud)
        #pts_road = self.point_cloud_filter_road.filter(pts_road)

        #image_reconstruction = converter.lidar_2_image(frame_data.point_cloud)

        #self.input_visu.show(frame_data)
        #self.pano_visu.show(frame_data.image_color, pano_image)
        #self.lidar_visu.show(pts_road)
        #self.projection_visu.show(image_reconstruction)
        #self.signal_visu.show(pano_image)

        #label_output = self.label_pano_tool.label(frame_count, pano_image)
        #self.label_visu.show(label_output)

        #plt.show()
        #mlab.show()

        #mlab.close()
        #plt.close()
        return pano_image
