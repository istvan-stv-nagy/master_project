from implementation.curbstone.road_edge_classifier import RoadEdgeClassifier
from implementation.datastructures.freespace_output import FreespaceOutput
from implementation.datastructures.pano_image import PanoImage
from implementation.dataset.kitti_dataset import KittiDataset, FrameData
import matplotlib.pyplot as plt
import numpy as np

from implementation.evaluation.pixel_evaluation import PixelEvaluation, PixelMetrics
from implementation.net.segnet import SegNet
from implementation.net.unet import UNET
from implementation.postprocessing.grid_occupancy import get_grid_occupancy
from implementation.utils.conversions import Converter
from implementation.utils.network_utils import *
from implementation.visu.curbstone_visu import CurbstoneVisu
from implementation.visu.evaluation_visu import EvaluationVisu
from implementation.visu.lidar_visu import LidarVisu
from implementation.visu.output_visu import OutputVisu
from implementation.visu.pano_visu import PanoVisu
from implementation.visu.workflow_visu import WorkFlowVisu

TESTING_IMAGE_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\image_2'
TESTING_CALIB_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\calib'
TESTING_VELO_DIR = r'E:\Storage\7 Master Thesis\dataset\velodyne\training\velodyne'
TESTING_GT_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\gt_image_2'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r'E:\Storage\7 Master Thesis\results\checkpoints\unet\checkpoint_acc9762_loss0.0462.pth.tar'


def main():
    dataset = KittiDataset(
        image_dir=TESTING_IMAGE_DIR,
        calib_dir=TESTING_CALIB_DIR,
        velo_dir=TESTING_VELO_DIR,
        gt_dir=TESTING_GT_DIR
    )

    model: UNET = load_checkpoint(model_path=MODEL_PATH, model=UNET(in_channels=1, out_channels=1), device=DEVICE)

    road_edge_classifier: RoadEdgeClassifier = RoadEdgeClassifier()

    workflow_visu = WorkFlowVisu(save_fig=False, dump_path=r'E:\Storage\7 Master Thesis\results\dumps\workflow_visu')
    lidar_visu = LidarVisu(y_range=(-20, 20), x_range=(0, 100))
    pano_visu = PanoVisu(save_fig=False, dump_path=r'E:\Storage\7 Master Thesis\results\dumps\pano_visu\distance')
    evaluation_visu = EvaluationVisu()
    output_visu = OutputVisu(save_fig=True, dump_path=r'E:\Storage\7 Master Thesis\results\dumps\freespace_polygon')
    curbstone_visu = CurbstoneVisu()

    pixel_evaluation = PixelEvaluation()

    for i in range(98, dataset.__len__()):
        # read frame
        frame_data: FrameData = dataset[i]
        # create converter object
        converter: Converter = Converter(calib=frame_data.calib)

        # create pano image
        pano: PanoImage = converter.lidar_2_pano(
            lidar_coords=frame_data.point_cloud,
            horizontal_fov=(-90, 90),
            vertical_fov=(-24.9, 2.0)
        )

        # run trained model
        prediction = run_model(np.array([[pano.y_road_img]]), model=model)

        # project masked prediction velo points to image
        projection_coordinates = converter.velo_2_image(pano.velo(mask=prediction))

        # freespace output
        freespace: FreespaceOutput = get_grid_occupancy(projection_coordinates.T, in_size=(375, 1242), grid_cell_size=(8, 8))

        # extract curb
        road_edges, edge_profiles = road_edge_classifier.run(prediction, pano)

        # run evaluations
        pixel_metrics: PixelMetrics = pixel_evaluation.run(freespace.mask, frame_data.gt_image)

        curbstone_visu.show(frame_data.image_color, prediction, road_edges, edge_profiles)
        # workflow_visu.show(index, frame_data.image_color, pano, prediction, projection_coordinates, freespace)
        # lidar_visu.show(frame_data.point_cloud, frame_data.point_cloud[0, :])
        # pano_visu.show(index, pano)
        # evaluation_visu.show(gt_image=frame_data.gt_image, prediction_image=freespace.mask)
        # output_visu.show(i, image=frame_data.image_color, freespace=freespace, fill=False)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
