import cv2
import numpy as np
import rasterio
from torch.utils.data import Dataset
import config
from src.utils import normalize


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
        vv_image_normalized = normalize(vv_image, config.mean_vv, config.std_vv)
        vh_image_normalized = normalize(vh_image, config.mean_vh, config.std_vh)

        if config.output_type == "semantic_map":
            semantic_map_path = df_row[f"semantic_map_prev_level"]
            if semantic_map_path:
                semantic_map = cv2.imread(semantic_map_path, 0) / 255.0
                input_image = np.dstack((vv_image_normalized, vh_image_normalized, semantic_map))
            else:
                dummy_channel = np.zeros_like(vv_image)
                input_image = np.dstack((vv_image_normalized, vh_image_normalized, dummy_channel))
        elif config.output_type == "softmax_prob":
            softmax_prob_path = df_row[f"softmax_prob_prev_level"]
            if softmax_prob_path:
                softmax_prob = np.load(softmax_prob_path)
                input_image = np.dstack((vv_image_normalized, vh_image_normalized, softmax_prob))
            else:
                dummy_channel = np.zeros_like(vv_image)
                input_image = np.dstack((vv_image_normalized, vh_image_normalized, dummy_channel))
        else:
            raise ValueError("Invalid output type to build one of input channels")

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
        if config.output_type == "semantic_map":
            semantic_map_path = df_row["semantic_map_prev_level"]
            if semantic_map_path:
                semantic_map = cv2.imread(semantic_map_path, 0) / 255.0
                input_image = np.dstack((sar_image, semantic_map))
            else:
                dummy_channel = np.zeros_like(sar_image[:, :, 0])
                input_image = np.dstack((sar_image, dummy_channel))
        elif config.output_type == "softmax_prob":
            softmax_prob_path = df_row[f"softmax_prob_prev_level"]
            if softmax_prob_path:
                softmax_prob = np.load(softmax_prob_path)
                input_image = np.dstack((sar_image, softmax_prob))
            else:
                dummy_channel = np.zeros_like(sar_image[:, :, 0])
                input_image = np.dstack((sar_image, dummy_channel))
        else:
            raise ValueError("Invalid output type to build one of input channels")

        mask = cv2.imread(mask_path, 0) / 255.0

        if self.transforms:
            transformed = self.transforms(image=input_image, mask=mask)
            input_image = transformed["image"]
            mask = transformed["mask"]

        # Calculate padding dimensions after stacking
        pad_height = 32 - (input_image.shape[0] % 32) if input_image.shape[0] % 32 != 0 else 0
        pad_width = 32 - (input_image.shape[1] % 32) if input_image.shape[1] % 32 != 0 else 0

        if pad_height > 0 or pad_width > 0:
            input_image = np.pad(input_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), 'constant')

        example["image"] = np.transpose(input_image, (2, 0, 1)).astype('float32')
        example["mask"] = mask.astype('int64')
        return example










