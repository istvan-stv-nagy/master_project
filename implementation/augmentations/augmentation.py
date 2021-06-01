import albumentations as A


class AugmentationFunctions:
    def __init__(self):
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=1.0)
            ]
        )

    def run(self, image, mask):
        augmentations = self.transform(image=image, mask=mask)
        augmented_image = augmentations["image"]
        augmented_mask = augmentations["mask"]
        return augmented_image, augmented_mask
