import cv2
import os
from torch.utils.data import Dataset
import utils
from PIL import Image


class ETCIDataset(Dataset):
    """
    Reference:
    https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a
    """
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

        # TODO: remove from training
        rgb_image = utils.grayscale_to_rgb(vv_image, vh_image)

        if self.split == "test":
            example["image"] = rgb_image.transpose((2, 0, 1))
        else:
            flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0
            water_mask = cv2.imread(df_row["water_label_path"], 0) / 255.0
            if self.transform:
                # augmented = self.transform(image=rgb_image, mask=flood_mask)
                augmented = self.transform(image=rgb_image, mask=water_mask)
                rgb_image = augmented["image"]
                # flood_mask = augmented["mask"]
                water_mask = augmented["mask"]

            # example["mask"] = flood_mask.astype('int64')
            example["image"] = rgb_image.transpose((2, 0, 1)).astype('float32')
            example["mask"] = water_mask.astype('int64')
        return example


class SN6Dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

        # get list of image and mask files
        self.image_files = sorted(os.listdir(os.path.join(self.data_dir, "images")))
        self.mask_files = sorted(os.listdir(os.path.join(self.data_dir, "masks")))

    def __getitem__(self, index):
        # load image and mask
        image_path = os.path.join(self.data_dir, "images", self.image_files[index])
        mask_path = os.path.join(self.data_dir, "masks", self.mask_files[index])
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

    def __len__(self):
        return len(self.image_files)



