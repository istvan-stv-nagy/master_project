from implementation.augmentations.augmentation import AugmentationFunctions
from implementation.dataset.segmentation_dataset import SegmentationDataset
from PIL import Image

IMAGE_DIR = r'E:\Storage\7 Master Thesis\dataset\semseg\train\image'
MASK_DIR = r'E:\Storage\7 Master Thesis\dataset\semseg\train\mask'
SAVE_IMAGE_DIR = r'E:\Storage\7 Master Thesis\dataset\semseg\train\augmented_image'
SAVE_MASK_DIR = r'E:\Storage\7 Master Thesis\dataset\semseg\train\augmented_mask'


def main():
    dataset = SegmentationDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, return_example_id=True)
    augmentation_function = AugmentationFunctions()
    for i in range(dataset.__len__()):
        image, mask, example_id = dataset[i]
        augmented_image, augmented_mask = augmentation_function.run(image, mask)
        augmented_image = Image.fromarray(augmented_image)
        augmented_mask = Image.fromarray(augmented_mask)
        augmented_image.save(SAVE_IMAGE_DIR + "\\aug_" + example_id + ".tif")
        augmented_mask.save(SAVE_MASK_DIR + "\\aug_" + example_id + ".tif")


if __name__ == '__main__':
    main()
