import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from implementation.datastructures.freespace_output import FreespaceOutput
from implementation.datastructures.pano_image import PanoImage
from implementation.evaluation.pixel_evaluation import PixelMetrics
from implementation.net.unet import UNET
from implementation.postprocessing.grid_occupancy import get_grid_occupancy
from implementation.utils.conversions import Converter
from implementation.utils.network_utils import *

VAL_IMG_DIR = r"G:\Steve\master\master_dataset\train\image_val"
VAL_MASK_DIR = r"G:\Steve\master\master_dataset\train\mask_val"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 514

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
    model: MODEL_TYPE = load_checkpoint(model_path=MODEL_PATH, model=MODEL_TYPE(in_channels=1, out_channels=1), device=DEVICE)

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=2.0
            ),
            ToTensorV2()
        ]
    )
    dataset = SegmentationDataset(
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        transform=val_transform
    )

    for i in dataset.__len__():
        # read frame
        image, mask = dataset[i]


if __name__ == '__main__':
    main()
