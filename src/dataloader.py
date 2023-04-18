import albumentations as A
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ETCIDataset, SN6Dataset
from sklearn.model_selection import train_test_split
from utils import (
    get_etci_df, cleanup_etci_data, cleanup_sn6_data, get_sn6_df, get_sn6_not_processed
)
import config


def get_loader(dataset_name):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(width=256, height=256)
    ])

    if dataset_name == "etci":
        train_df, val_df = split_etci_data()
        train_dataset = ETCIDataset(dataframe=train_df, split="train", transform=transform)
        val_dataset = ETCIDataset(dataframe=val_df, split="valid", transform=None)
    else:
        train_df, val_df = split_sn6_data()
        train_dataset = SN6Dataset(dataframe=train_df, split="train", transform=transform)
        val_dataset = SN6Dataset(dataframe=val_df, split="valid", transform=None)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader


def split_etci_data():
    regions = ["nebraska", "northal", "bangladesh"]
    # randomly choose one for the validation set and leave the rest for training
    validation_region = np.random.choice(regions, 1)[0]
    regions.remove(validation_region)
    original_df = get_etci_df(config.train_dir, split="train")
    print(f"Original:{original_df.shape}")
    original_df = cleanup_etci_data(original_df)
    print(f"Cleaned up:{original_df.shape}")
    train_df = original_df[original_df['region'] != validation_region]
    val_df = original_df[original_df['region'] == validation_region]
    print(f"Split into train:{train_df.shape} and valid:{val_df.shape}")
    return train_df, val_df


def split_sn6_data():
    original_df = get_sn6_df(split="train")
    print(f"Original:{original_df.shape}")
    summary_df = pd.read_csv(config.sn6_summary_datapath)
    image_ids = summary_df.ImageId.unique()
    not_processed = get_sn6_not_processed(mask_train_dir=config.mask_train_dir, image_ids=image_ids)
    original_df = cleanup_sn6_data(original_df, not_processed)
    print(f"Cleaned up:{original_df.shape}")
    train_df, val_df = train_test_split(original_df, test_size=0.2)
    print(f"Split into train:{train_df.shape} and valid:{val_df.shape}")
    return train_df, val_df