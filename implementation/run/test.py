from implementation.datastructures.pano_image import PanoImage
from implementation.dataset.kitti_dataset import KittiDataset, FrameData
import matplotlib.pyplot as plt
import numpy as np

from implementation.net.unet import UNET
from implementation.postprocessing.grid_occupancy import get_grid_occupancy
from implementation.utils.conversions import Converter
from implementation.utils.network_utils import *
from implementation.visu.lidar_visu import LidarVisu
from implementation.visu.pano_visu import PanoVisu
from implementation.visu.workflow_visu import WorkFlowVisu

TESTING_IMAGE_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\image_2'
TESTING_CALIB_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\training\calib'
TESTING_VELO_DIR = r'E:\Storage\7 Master Thesis\dataset\velodyne\training\velodyne'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r'E:\Storage\7 Master Thesis\results\checkpoints\checkpoint_acc9762_loss0.0462.pth.tar'


def run_frame(index,
              frame_data,
              model,
              workflow_visu: WorkFlowVisu,
              lidar_visu: LidarVisu,
              pano_visu: PanoVisu):
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

    # occupancy grid mapping
    occupancy = get_grid_occupancy(projection_coordinates.T, in_size=(375, 1242), grid_cell_size=(8, 8))

    workflow_visu.show(index, frame_data.image_color, pano, prediction, projection_coordinates, occupancy)
    lidar_visu.show(frame_data.point_cloud, frame_data.point_cloud[0, :])
    pano_visu.show(index, pano)
    plt.show()
    plt.close()


def main():
    dataset = KittiDataset(
        image_dir=TESTING_IMAGE_DIR,
        calib_dir=TESTING_CALIB_DIR,
        velo_dir=TESTING_VELO_DIR
    )

    model: UNET = load_checkpoint(model_path=MODEL_PATH, model=UNET(in_channels=1, out_channels=1), device=DEVICE)

    workflow_visu = WorkFlowVisu(save_fig=False, dump_path=r'E:\Storage\7 Master Thesis\results\dumps\workflow_visu')
    lidar_visu = LidarVisu(y_range=(-20, 20), x_range=(0, 100))
    pano_visu = PanoVisu(save_fig=True, dump_path=r'E:\Storage\7 Master Thesis\results\dumps\pano_visu\depth')

    for i in range(dataset.__len__()):
        # read frame
        frame_data: FrameData = dataset[i]
        run_frame(i, frame_data, model, workflow_visu, lidar_visu, pano_visu)


if __name__ == '__main__':
    main()
