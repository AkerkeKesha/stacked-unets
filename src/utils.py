import os.path
from glob import glob
import pathlib
import os
import numpy as np
import pandas as pd
import cv2
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename="single_Unet_training.log",
                             filemode="w",
                             format="%(name)s - %(levelname)s - %(message)s",)


def get_logging():
    # TODO: make sure logger works
    return logging.getLogger()


def grayscale_to_rgb(vv_image, vh_image):
    # TODO: make sure NN does not use clipped data
    ratio_image = np.clip(np.nan_to_num(vh_image/(vv_image + 1e-6), 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1-ratio_image), axis=2)
    return rgb_image


def get_etci_df(dirname, split):
    vv_image_paths = sorted(glob(dirname + '/*/*/vv/*.png', recursive=True))
    vh_image_paths, flood_label_paths, water_body_label_paths, region_names = [], [], [], []
    for i in range(len(vv_image_paths)):
        vv_image_path = pathlib.PurePath(vv_image_paths[i])
        region_dirname = vv_image_path.parent.parent.parent

        vh_image_filename = str(vv_image_path).replace("vv", "vh")
        vh_image_path = os.path.join(dirname, region_dirname, "tiles", "vh", vh_image_filename)
        vh_image_paths.append(vh_image_path)
        if split == "test":
            flood_label_paths.append(np.NaN)
        else:
            flood_label_filename = os.path.basename(vv_image_path).replace("_vv", "")
            flood_label_path = os.path.join(dirname, region_dirname, "tiles", "flood_label", flood_label_filename)
            flood_label_paths.append(flood_label_path)

        water_label_filename = os.path.basename(vv_image_path).replace("_vv", "")
        water_body_label_path = os.path.join(dirname, region_dirname, "tiles", "water_body_label", water_label_filename)
        water_body_label_paths.append(water_body_label_path)

        region_names.append(os.path.basename(vv_image_paths[i]).split("_")[0])

    paths = {
        "vv_image_path": vv_image_paths,
        "vh_image_path": vh_image_paths,
        "flood_label_path": flood_label_paths,
        "water_body_label_path": water_body_label_paths,
        "region": region_names,
    }
    return pd.DataFrame(paths)


def visualize_image_and_masks(df_row, figure_size=(25, 15)):
    vv_image_path = df_row['vv_image_path']
    vh_image_path = df_row['vh_image_path']
    flood_label_path = df_row['flood_label_path']
    water_body_label_path = df_row['water_body_label_path']

    rgb_filename = os.path.basename(vv_image_path)
    vv_image = cv2.imread(vv_image_path, 0) / 255.0
    vh_image = cv2.imread(vh_image_path, 0) / 255.0
    rgb_image = grayscale_to_rgb(vv_image, vh_image)

    water_body_label_image = cv2.imread(water_body_label_path, 0) / 255.0

    plt.figure(figsize=figure_size)

    if df_row.isnull().sum() > 0:
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title(rgb_filename)

        # plot water body mask
        plt.subplot(1, 2, 2)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')
    else:
        flood_label_image = cv2.imread(flood_label_path, 0) / 255.0

        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title(rgb_filename)

        # plot flood label mask
        plt.subplot(1, 3, 2)
        plt.imshow(flood_label_image)
        plt.title('Flood mask')

        # plot water body mask
        plt.subplot(1, 3, 3)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')


def visualize_prediction(df_row, prediction, figure_size=(25, 15)):
    vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
    vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0
    rgb_input = grayscale_to_rgb(vv_image, vh_image)

    plt.figure(figsize=figure_size)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_input)
    plt.title('RGB w/ result')
    plt.subplot(1, 2, 2)
    plt.imshow(prediction)
    plt.title('Prediction')


# def get_sn6_df(dirname, split):
#     image_paths, label_paths = [], []
#     for i in range(len(image_paths)):
#         image_path =
#         image_paths.append(image_path)
#         if split == "test":
#             label_paths.append(np.NaN)
#         else:
#             label_path =
#             label_paths.append(label_path)
#
#     paths = {
#         "image_path": image_paths,
#         "label_path": label_paths,
#     }
#     return pd.DataFrame(paths)

