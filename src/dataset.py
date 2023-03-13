"""
Reference:
https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a
"""
import cv2
from torch.utils.data import Dataset
import utils


class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.dataset = dataframe
        self.split = split
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}
        df_row = self.dataset.iloc[index]

        vv_image = cv2.imread(df_row["vv_image_path"], 0) / 255.0
        vh_image = cv2.imread(df_row["vh_image_path"], 0) / 255.0

        rgb_image = utils.grayscale_to_rgb(vv_image, vh_image)

        if self.split == "test":
            example["image"] = rgb_image.transpose((2, 0, 1))
        else:
            flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0
            if self.transform:
                augmented = self.transform(image=rgb_image, mask=flood_mask)
                rgb_image = augmented["image"]
                flood_mask = augmented["mask"]
            example["image"] = rgb_image.transpose((2, 0, 1)).astype('float32')
            example["mask"] = flood_mask.astype('int64')

        return example



