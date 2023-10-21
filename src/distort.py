import os
import cv2
import copy
import numpy as np
import torch
from tqdm import tqdm
import config
from src.model import UNet
from src.evaluate import IntersectionOverUnion


def load_labels(labels_dir, target_level):
    """
    @param labels_dir: here prediction images are stored in f"{labels_dir}/semantic_map_level_0_image.png format
    @param target_level: the level of interest
    @return: list of semantic maps of this target level
    """
    collected_outputs = []
    for prediction_image_name in os.listdir(labels_dir):
        if f"{config.output_type}_level_{target_level}_" in prediction_image_name:
            prediction_path = os.path.join(labels_dir, prediction_image_name)
            if config.output_type == "semantic_map":
                label_data = cv2.imread(prediction_path, 0) / 255.0
            elif config.output_type == "softmax_prob":
                label_data = np.load(prediction_path)
            else:
                raise ValueError("Invalid output_type. Should be 'semantic_map' or 'softmax_prob'")
            collected_outputs.append(label_data)
    return collected_outputs


def distort_maps(input_maps, proportion=1.0, num_classes=2, make_copy=True):
    """
    Distort the proportion of semantic maps with random values
    """
    distorted_maps = []
    for map_ in input_maps:
        if make_copy:
            map_ = copy.deepcopy(map_)
        num_indices_to_distort = int(np.prod(map_.shape) * proportion)  # e.g. distort 100% of the map
        indices = np.random.choice(np.prod(map_.shape), num_indices_to_distort, replace=False)

        # Generate random values between [0, num_classes-1] for each selected index
        random_values = np.random.randint(0, num_classes, num_indices_to_distort)
        map_ = map_.flatten()
        map_[indices] = random_values
        map_ = map_.reshape(map_.shape)

        distorted_maps.append(map_)

    return distorted_maps


def predict_with_distortion(test_loader, df_test, run=0, level=0, distorted_semantic_maps=None):
    if distorted_semantic_maps is None:
        raise ValueError("Distorted semantic maps must be provided.")
    run_key = f"run{run}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=config.num_channels)
    model.load_state_dict(torch.load(f"{config.output_dir}/{run_key}_level{level}_unet_{config.dataset}.pt"))
    model.to(device)
    model.eval()

    final_predictions, true_labels = [], []
    iou_metric = IntersectionOverUnion(num_classes=2)
    distorted_semantic_maps_iter = iter(distorted_semantic_maps)

    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                true_masks = batch["mask"].numpy()
                for idx, (image, true_mask) in enumerate(zip(batch["image"], true_masks)):
                    distorted_map = next(distorted_semantic_maps_iter).to(device)

                    # Replace the semantic map (third channel) with the distorted one
                    image[2, :, :] = distorted_map.view(256, 256)
                    image = image.unsqueeze(0).to(device)

                    pred = model(image)
                    iou_metric.update(pred.detach().cpu().numpy(), true_mask)

                    class_label = pred.argmax(dim=1)
                    class_label = class_label.detach().cpu().numpy()
                    final_predictions.append(class_label.astype("uint8"))

    except Exception as te:
        print(f"An exception occurred during inference: {te}")

    mean_iou = iou_metric.mean_iou()
    print(f"Mean IoU for the test dataset with distorted semantic maps: {mean_iou}")
    final_predictions = np.concatenate(final_predictions, axis=0)

    return final_predictions, df_test, mean_iou
