import csv

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
MODEL_PATH = r'E:\Storage\7 Master Thesis\results\models\unet\unet_all_used\checkpoint_acc9762_loss0.0462.pth.tar'
#MODEL_PATH = r'D:\master_dataset\models\unet\34unet_checkpoint.pth.tar'
MODEL_TYPE = UNET

testing_names = [
    "um_000000", "um_000014", "um_000014", "um_000029", "um_000030", "um_000044", "um_000045", "um_000059", "um_000060", "um_000074", "um_000075", "um_000089", "um_000090",
    "umm_000009", "umm_000010", "umm_000024", "umm_000025", "umm_000039", "umm_000040", "umm_000054", "umm_000055", "umm_000069", "umm_000070", "umm_000084", "umm_000085", "umm_000090",
    "uu_000008", "uu_000009", "uu_000023", "uu_000024", "uu_000038", "uu_000039", "uu_000053", "uu_000054", "uu_000068", "uu_000069", "uu_000083", "uu_000084"
]

testing_ids = [9,10,24,25,39,40,54,55,69,70,84,85,90,96,110,125,126,140,141,155,156,170,171,185,186,199,200,214,215,229,230,244,245,259,260,274,275]

def main():
    f = open(r'E:\Storage\7 Master Thesis\results\models\unet\unet_all_used\eval.csv', "w", newline='')
    writer = csv.writer(f, delimiter=",")
    writer.writerow(PixelMetrics.names())
    kitti_dataset = KittiDataset(
        image_dir=TESTING_IMAGE_DIR,
        calib_dir=TESTING_CALIB_DIR,
        velo_dir=TESTING_VELO_DIR,
        gt_dir=TESTING_GT_DIR,
        return_name=True
    )

    model: MODEL_TYPE = load_checkpoint(model_path=MODEL_PATH, model=MODEL_TYPE(in_channels=1, out_channels=1), device=DEVICE)

    evaluation_visu = EvaluationVisu()

    pixel_evaluation = PixelEvaluation()

    xs = np.array([])
    for i in testing_ids:
        print("Running scene ", i)
        # read frame
        frame_data, name = kitti_dataset[i]
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

        xx, yy, zz = pano.velo(mask=prediction)
        xs = np.concatenate((xs, xx))

        # freespace output
        #freespace: FreespaceOutput = get_grid_occupancy(projection_coordinates.T, in_size=(375, 1242), grid_cell_size=(8, 8))

        # run evaluations
        #if freespace.mask.shape == frame_data.gt_image.shape:
        #    pixel_metrics: PixelMetrics = pixel_evaluation.run(freespace.mask, frame_data.gt_image)
        #    writer.writerow(pixel_metrics.values())

        # evaluation_visu.show(gt_image=frame_data.gt_image, prediction_image=freespace.mask)
        # plt.show()
        # plt.close()

    plt.figure()
    xs = xs[xs > 19.99]
    plt.hist(xs, bins=np.arange(20, 51, 1), density=True)
    plt.show()

    f.close()

if __name__ == '__main__':
    main()
