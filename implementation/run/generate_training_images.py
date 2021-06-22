from implementation.dataset.kitti_dataset import KittiDataset
from implementation.datastructures.pano_image import PanoImage
from implementation.utils.conversions import Converter
import numpy as np
import os
import torch

TRAINING_IMAGE_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\testing\image_2'
TRAINING_CALIB_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\testing\calib'
TRAINING_VELO_DIR = r'E:\Storage\7 Master Thesis\dataset\velodyne\testing\velodyne'
TRAINING_GT_DIR = r'E:\Storage\7 Master Thesis\dataset\data_road\testing\gt_image_2'

EXAMPLES_DUMP_PATH = r'E:\Storage\7 Master Thesis\dataset\semseg\dataset_roadXYZ_valid\image'
TESTING = True


def main():
    dataset = KittiDataset(
        image_dir=TRAINING_IMAGE_DIR,
        calib_dir=TRAINING_CALIB_DIR,
        velo_dir=TRAINING_VELO_DIR,
        gt_dir=TRAINING_GT_DIR,
        return_name=True
    )
    for i in range(dataset.__len__()):
        frame_data, name = dataset[i]
        converter: Converter = Converter(calib=frame_data.calib)
        pano: PanoImage = converter.lidar_2_pano(
            lidar_coords=frame_data.point_cloud,
            horizontal_fov=(-90, 90),
            vertical_fov=(-24.9, 2.0)
        )
        x = np.array([pano.x_road_img, pano.y_road_img, pano.z_road_img])

        names = name.split('_')
        if names[0] == 'um' and not TESTING:
            prefix = ''
        else:
            if TESTING:
                prefix = names[0] + 't'
            else:
                prefix = names[0]
        suffix = str(int(names[1]))
        file_name = prefix + suffix

        np.save(os.path.join(EXAMPLES_DUMP_PATH, file_name + ".npy"), x)


def test():
    x = np.load(r'E:\Storage\7 Master Thesis\dataset\semseg\train_roadXYZ\0.npy')
    x = torch.from_numpy(x)
    print(x)


def flip_numpy_array_from_folder(folder, out_folder):
    examples = os.listdir(folder)
    for example in examples:
        x = np.load(os.path.join(folder, example))
        x = np.flip(x, 2)
        np.save(os.path.join(out_folder, "aug_" + example), x)


if __name__ == '__main__':
    in_path = r'E:\Storage\7 Master Thesis\dataset\semseg\dataset_roadXYZ_valid\image'
    out_path = r'E:\Storage\7 Master Thesis\dataset\semseg\dataset_roadXYZ_valid\image_augmented'
    #main()
    flip_numpy_array_from_folder(in_path, out_path)
