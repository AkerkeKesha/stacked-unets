from glob import glob
from typing import List
import pathlib
import os
import numpy as np
import pandas as pd
import cv2
import config
import matplotlib.pyplot as plt

# logging.basicConfig(filename="single_Unet_training.log",
#                              filemode="w",
#                              format="%(name)s - %(levelname)s - %(message)s",)
#
#
# def get_logging():
#     # TODO: make sure logger works
#     return logging.getLogger()


def grayscale_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image/(vv_image + 1e-6), 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1-ratio_image), axis=2)
    return rgb_image


def sar_to_grayscale(vv_image, vh_image):
    gray_image = (vv_image + vh_image) / 2
    gray_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image))
    return gray_image


def get_etci_df(dirname, split):
    vv_image_paths = sorted(glob(dirname + '/*/*/vv/*.png', recursive=True))
    vh_image_paths, flood_label_paths, water_body_label_paths, region_names, semantic_map_paths = [], [], [], [], []
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
        semantic_map_paths.append("")

    paths = {
        "vv_image_path": vv_image_paths,
        "vh_image_path": vh_image_paths,
        "flood_label_path": flood_label_paths,
        "water_body_label_path": water_body_label_paths,
        "region": region_names,
        "semantic_map_prev_level": semantic_map_paths,
    }
    return pd.DataFrame(paths)


def cleanup_etci_data(df):
    noisy_points = []
    for i, image_path in enumerate(df['vv_image_path'].tolist()):
        image = cv2.imread(image_path, 0)
        image_values = list(np.unique(image))
        binary_value_check = (image_values == [0, 255]) or (image_values == [0]) or (image_values == [255])
        if binary_value_check:
            noisy_points.append(i)
    filtered_df = df.drop(df.index[noisy_points])
    return filtered_df


def plot_single_prediction(image_name, semantic_map_path, output_dir, figure_size=(6, 6)):
    if not os.path.exists(semantic_map_path):
        raise FileNotFoundError(f"File does not exist: {semantic_map_path}")
    semantic_map = cv2.imread(semantic_map_path, 0)
    if semantic_map is None or semantic_map.size == 0:
        raise FileNotFoundError(f"No file found or unable to read the file at: {semantic_map_path}")
    plt.figure(figsize=figure_size)
    plt.imshow(semantic_map)
    plt.axis('off')
    # output_path = os.path.join(output_dir, f"prediction_{image_name}.png")
    # plt.savefig(output_path, bbox_inches='tight')
    plt.close()


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

        plt.subplot(1, 2, 2)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')
    else:
        flood_label_image = cv2.imread(flood_label_path, 0) / 255.0

        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title(rgb_filename)

        plt.subplot(1, 3, 2)
        plt.imshow(flood_label_image)
        plt.title('Flood mask')

        plt.subplot(1, 3, 3)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')


def find_prediction_image(searched_value, df):
    mask = df['vv_image_path'].str.endswith(searched_value)
    indices = df.loc[mask].index[0]
    if indices.size > 0:
        return indices[0]
    else:
        print(f"No match found for {searched_value} in vv_image_path column")
        return None


def visualize_prediction(prediction_image_name, original_df, figure_size=(25, 15)):
    index = find_prediction_image(f'{prediction_image_name}_vv.png', original_df)
    if index is not None:
        df_row = original_df.iloc[index]

        vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
        vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0
        rgb_input = grayscale_to_rgb(vv_image, vh_image)

        water_body_label_path = df_row['water_body_label_path']
        water_body_label_image = cv2.imread(water_body_label_path, 0) / 255.0

        flood_label_path = df_row['flood_label_path']
        flood_label_image = cv2.imread(flood_label_path, 0) / 255.0

        image_id = os.path.basename(df_row['vv_image_path']).split('.')[0]

        prediction_path = df_row["semantic_map_prev_level"]
        if not os.path.exists(prediction_path):
            raise FileNotFoundError(f"File does not exist: {prediction_path}")
        if prediction_path is None or prediction_path.size == 0:
            raise FileNotFoundError(f"Unable to read the image file: {prediction_image_name}")

        prediction = cv2.imread(prediction_path, 0)
        if prediction is None:
            raise FileNotFoundError(f"Unable to load the image: {prediction_image_name}")

        prediction /= 255.0

        plt.figure(figsize=figure_size)

        plt.subplot(1, 4, 1)
        plt.imshow(rgb_input)
        plt.title(f'{image_id}')

        plt.subplot(1, 4, 2)
        plt.imshow(water_body_label_image)
        plt.title('Water body mask')

        plt.subplot(1, 4, 3)
        plt.imshow(flood_label_image)
        plt.title('Flood mask')

        plt.subplot(1, 4, 4)
        plt.imshow(prediction)
        plt.title(f'Prediction {prediction_image_name}')
        plt.show()
    else:
        print(f"Skipping visualization for {prediction_image_name} due to missing data")


def get_image_name_from_path(image_path: str):
    """
    Extracts the image name from the file path.
    Example: 'path/to/some_parts_of_image_vv.png' -> 'image'
    """
    base_name = os.path.basename(image_path)
    # Split the base_name by '_', keep all but the last element, and join them back
    image_name_parts = base_name.split('_')[:-1]
    image_name = '_'.join(image_name_parts)
    return image_name


def store_semantic_maps(df: pd.DataFrame, n_levels: int, semantic_maps: List):
    """
    Stores the generated semantic maps in the DataFrame.
    """
    for i, (_, df_row) in enumerate(df.iterrows()):
        image_path = df_row["vv_image_path"]
        image_name = get_image_name_from_path(image_path)
        semantic_map = semantic_maps[i][0]  # access the first (and only) element in each item
        semantic_map_path = f"{config.output_dir}/{config.dataset}_labels/semantic_map_level_{n_levels}_image_{image_name}.png"
        cv2.imwrite(semantic_map_path, semantic_map * 255)
        df.at[_, f"semantic_map_prev_level"] = semantic_map_path


def get_sn6_df(split, mode="SAR-Intensity"):
    image_ids, image_paths, mask_paths = [], [], []
    if split == "train":
        summary_df = pd.read_csv(config.sn6_summary_datapath)
        image_ids = summary_df.ImageId.unique()
        for image_id in image_ids:
            image_path = f'{config.train_dir}/{mode}/SN6_Train_AOI_11_Rotterdam_{mode}_{image_id}.tif'
            image_paths.append(image_path)
            mask_path = f'{config.mask_train_dir}/SN6_Train_AOI_11_Rotterdam_{mode}_{image_id}.png'
            mask_paths.append(mask_path)
    elif split == "test":
        image_ids = get_sn6_test_image_ids(test_dir=config.test_dir)
        for image_id in image_ids:
            image_path = f'{config.test_dir}/{mode}/SN6_Test_Public_AOI_11_Rotterdam_{mode}_{image_id}.tif'
            image_paths.append(image_path)
            mask_paths.append(np.NaN)
    paths = {
        "image_id": image_ids,
        "image_path": image_paths,
        "mask_path": mask_paths,
    }
    return pd.DataFrame(paths)


def cleanup_sn6_data(df, not_processed):
    cleaned_df = df[~df['image_id'].isin(not_processed)]
    return cleaned_df


def get_sn6_not_processed(mask_train_dir, image_ids):
    not_processed = []
    for image_id in image_ids:
        mode = 'SAR-Intensity'
        out_filename = f'SN6_Train_AOI_11_Rotterdam_{mode}_{image_id}.png'
        mask_path = os.path.join(mask_train_dir, out_filename)
        if not os.path.exists(mask_path):
            not_processed.append(image_id)
    return not_processed


def get_sn6_test_image_ids(test_dir):
    search_pattern = os.path.join(test_dir, 'SAR-Intensity', 'SN6_Test_Public_AOI_11_Rotterdam_*.tif')
    file_paths = glob(search_pattern)
    image_ids = [os.path.basename(file_path).replace('SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_', '')
                 .replace('.tif', '') for file_path in file_paths]
    return image_ids



