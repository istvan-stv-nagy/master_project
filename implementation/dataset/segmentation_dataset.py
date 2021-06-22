import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, return_example_id=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.return_example_id = return_example_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item])
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        if self.return_example_id:
            return image, mask, self.images[item].split('.')[0]
        return image, mask


class SegmentationDatasetRoadXYZ(Dataset):
    def __init__(self, pano_dir, mask_dir):
        self.pano_dir = pano_dir
        self.mask_dir = mask_dir
        self.examples = os.listdir(pano_dir)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example_path = os.path.join(self.pano_dir, self.examples[item])
        mask_path = os.path.join(self.mask_dir, self.examples[item].replace(".npy", ".tif"))
        example = np.load(example_path)
        mask = np.array(Image.open(mask_path))
        return np.asarray(example, np.float32), np.asarray(mask, np.float32)
