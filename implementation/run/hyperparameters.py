import os.path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from implementation.net.segnet import SegNet
from implementation.net.unet import UNET
from implementation.utils.network_utils import *

# hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 514
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"G:\Steve\master\master_dataset\train\image"
TRAIN_MASK_DIR = r"G:\Steve\master\master_dataset\train\mask"
VAL_IMG_DIR = r"G:\Steve\master\master_dataset\train\image_val"
VAL_MASK_DIR = r"G:\Steve\master\master_dataset\train\mask_val"

BATCH_SIZES = [8]
LEARNING_RATES = [0.001]


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

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)

    for batch_size in BATCH_SIZES:
        for lr in LEARNING_RATES:
            step = 0
            writer = SummaryWriter(f'runs/unet/bsize{batch_size}_LR{lr}')
            train_loader, val_loader = get_loaders(
                TRAIN_IMG_DIR,
                TRAIN_MASK_DIR,
                VAL_IMG_DIR,
                VAL_MASK_DIR,
                batch_size,
                train_transform,
                val_transform,
                NUM_WORKERS,
                PIN_MEMORY
            )
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            scaler = torch.cuda.amp.GradScaler()

            for epoch in range(NUM_EPOCHS):
                loop = tqdm(train_loader)
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

                # check accuracy
                acc = check_accuracy(val_loader, model, device=DEVICE)
                writer.add_scalar('loss', loss, global_step=step)
                writer.add_scalar('accuracy', acc, global_step=step)
                step += 1


if __name__ == '__main__':
    main()
