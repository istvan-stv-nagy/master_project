import os.path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from implementation.net.segnet import SegNet
from implementation.net.unet import UNET
from implementation.utils.network_utils import *

# hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 35
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 514
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"G:\Steve\master\master_dataset\semseg\dataset_roadY\image"
TRAIN_MASK_DIR = r"G:\Steve\master\master_dataset\semseg\masks\mask"
VAL_IMG_DIR = r"G:\Steve\master\master_dataset\semseg\dataset_roadY\image_val"
VAL_MASK_DIR = r"G:\Steve\master\master_dataset\semseg\masks\mask_val"


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=2.0
            ),
            ToTensorV2()
        ]
    )

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

    #model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    model = SegNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=os.path.join(r'G:\Steve\master\master_dataset\semseg\dataset_roadY\train_results\segnet', str(epoch) + "unet_checkpoint.pth.tar"))

        # check accuracy
        print("Running epoch:", epoch)
        check_accuracy(val_loader, model, device=DEVICE)

        # print examples
        # save_predictions_as_imgs(val_loader, model, folder=r'G:\Steve\master\checkpoints\segnet\saved_images', device=DEVICE)


if __name__ == '__main__':
    main()
