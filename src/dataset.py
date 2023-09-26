import cv2
import numpy as np
import rasterio
from torch.utils.data import Dataset
from torch.nn.functional import pad


class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.dataset = dataframe
        self.split = split
        self.transform = transform
        self.indices = list(range(len(dataframe)))

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}
        df_row = self.dataset.iloc[index]
        vv_image = cv2.imread(df_row["vv_image_path"], 0) / 255.0
        vh_image = cv2.imread(df_row["vh_image_path"], 0) / 255.0

        semantic_map_path = df_row[f"semantic_map_prev_level"]
        if semantic_map_path:
            semantic_map = cv2.imread(semantic_map_path, 0) / 255.0
            input_image = np.dstack((vv_image, vh_image, semantic_map))
        else:
            dummy_channel = np.zeros_like(vv_image)
            input_image = np.dstack((vv_image, vh_image, dummy_channel))

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
            sar_image = src.read()
        sar_image /= 255.0
        sar_image = np.transpose(sar_image, (1, 2, 0))

        pad_height = 32 - (sar_image.shape[1] % 32)
        pad_width = 32 - (sar_image.shape[2] % 32)

        semantic_map_path = df_row["semantic_map_prev_level"]
        if semantic_map_path:
            semantic_map = cv2.imread(semantic_map_path, 0) / 255.0
            input_image = np.dstack((sar_image, semantic_map))
        else:
            dummy_channel = np.zeros_like(sar_image[:, :, 0])
            input_image = np.dstack((sar_image, dummy_channel))

        mask = cv2.imread(mask_path, 0) / 255.0

        if self.transforms:
            transformed = self.transforms(image=input_image, mask=mask)
            input_image = transformed["image"]
            mask = transformed["mask"]

        # Apply padding to the stacked input and mask
        input_image = pad(input_image, (0, pad_width, 0, pad_height), mode='constant', value=0)
        mask = pad(mask, (0, pad_width, 0, pad_height), mode='constant', value=0)

        example["image"] = np.transpose(input_image, (2, 0, 1)).astype('float32')
        example["mask"] = mask.astype('int64')
        return example










