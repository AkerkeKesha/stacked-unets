import cv2
import numpy as np
import rasterio
from torch.utils.data import Dataset


class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.dataset = dataframe
        self.split = split
        self.transform = transform
        self.indices = dataframe.index

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}
        df_row = self.dataset.iloc[index]
        vv_image = cv2.imread(df_row["vv_image_path"], 0) / 255.0
        vh_image = cv2.imread(df_row["vh_image_path"], 0) / 255.0
        input_image = np.dstack((vv_image, vh_image))
        flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0
        if self.transform:
            augmented = self.transform(image=input_image, mask=flood_mask)
            input_image = augmented["image"]
            flood_mask = augmented["mask"]
        example["mask"] = flood_mask.astype('int64')
        example["image"] = np.transpose(input_image, (2, 0, 1)).astype('float32')
        return example


class SN6Dataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.dataframe = dataframe
        self.transforms = transform
        self.split = split

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        example = {}
        df_row = self.dataframe.iloc[index]
        image_path = df_row["image_path"]
        mask_path = df_row["mask_path"]

        with rasterio.open(image_path) as src:
            channels = src.read()
        max_value = channels.max()
        channels /= max_value
        sar_image = np.transpose(channels, (1, 2, 0))
        if self.split == "test":
            example["image"] = np.transpose(sar_image, (2, 0, 1)).astype('float32')
        else:
            mask = cv2.imread(mask_path, 0)
            if self.transforms:
                transformed = self.transforms(image=sar_image, mask=mask)
                sar_image = transformed["image"]
                mask = transformed["mask"]

            example["image"] = np.transpose(sar_image, (2, 0, 1)).astype('float32')
            example["mask"] = mask.astype('int64')
        return example









