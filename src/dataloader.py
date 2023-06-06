import albumentations as A
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ETCIDataset, SN6Dataset
from sklearn.model_selection import train_test_split
from utils import (
    get_etci_df, cleanup_etci_data,
    cleanup_sn6_data, get_sn6_df, get_sn6_not_processed
)
import config


def get_loader(dataset_name, train_df, val_df, test_df):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(width=256, height=256)
    ])

    if dataset_name == "etci":
        train_dataset = ETCIDataset(dataframe=train_df, split="train", transform=transform)
        val_dataset = ETCIDataset(dataframe=val_df, split="valid", transform=None)
        test_dataset = ETCIDataset(dataframe=test_df, split="test", transform=None)
    else:
        train_dataset = SN6Dataset(dataframe=train_df, split="train", transform=transform)
        val_dataset = SN6Dataset(dataframe=val_df, split="valid", transform=None)
        test_dataset = SN6Dataset(dataframe=test_df, split="test", transform=None)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader


def split_etci_data(test_size=0.1, val_size=0.1):
    original_df = get_etci_df(config.train_dir, split="train")
    original_df = cleanup_etci_data(original_df)
    original_df = original_df.reset_index(drop=True)

    train_df, temp_df = train_test_split(original_df, test_size=(test_size + val_size), random_state=42)
    adjusted_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - adjusted_val_size), random_state=42)
    print(f"Split into train:{train_df.shape}, validation:{val_df.shape}, and test:{test_df.shape}")
    return original_df, train_df, val_df, test_df


def split_sn6_data(test_size=0.1, val_size=0.1):
    original_df = get_sn6_df(split="train")
    summary_df = pd.read_csv(config.sn6_summary_datapath)
    image_ids = summary_df.ImageId.unique()
    not_processed = get_sn6_not_processed(mask_train_dir=config.mask_train_dir, image_ids=image_ids)
    original_df = cleanup_sn6_data(original_df, not_processed)

    train_df, temp_df = train_test_split(original_df, test_size=test_size + val_size, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=42)
    print(f"Split into train:{train_df.shape}, validation:{val_df.shape}, and test:{test_df.shape}")
    return original_df, train_df, val_df, test_df