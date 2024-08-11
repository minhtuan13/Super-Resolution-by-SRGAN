import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 8
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3


def dynamic_normalize(image, **kwargs):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    normalized_image = (image - mean) / (std + 1e-7)
    return normalized_image

highres_transform = A.Compose(
    [
        A.Lambda(image=dynamic_normalize),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Lambda(image=dynamic_normalize),
        ToTensorV2(),
    ]
)


both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Lambda(image=dynamic_normalize),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Lambda(image=dynamic_normalize),
        ToTensorV2(),
    ]
)