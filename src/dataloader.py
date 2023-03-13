import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from dataset import ETCIDataset
from utils import create_df, get_logging
import config


def get_loader():
    regions = ["nebraska", "northal", "bangladesh"]

    # randomly choose one for the validation set and leave the rest for training
    validation_region = np.random.choice(regions, 1)[0]
    regions.remove(validation_region)

    original_df = create_df(config.train_dir, split="train")
    train_df = original_df[original_df['region'] != validation_region]
    val_df = original_df[original_df['region'] == validation_region]

    logger = get_logging()
    logger.info(f"Split into train:{train_df.shape} and valid:{val_df.shape}")

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(width=256, height=256)
    ])

    train_dataset = ETCIDataset(dataframe=train_df, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_dataset = ETCIDataset(dataframe=val_df, split="valid", transform=None)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader