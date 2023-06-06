import torch
import numpy as np
from tqdm import tqdm
from model import StackedUNet, basic_unet
from evaluate import IntersectionOverUnion
from utils import store_semantic_maps
import config


def predict(test_loader, df_test, n_levels=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StackedUNet(n_levels=n_levels, base_model=basic_unet)
    model.load_state_dict(torch.load(f"{config.output_dir}/level_{n_levels}_unet_{config.dataset}.pt"))
    model.to(device)
    model.eval()

    final_predictions, true_labels = [], []
    iou_metric = IntersectionOverUnion(num_classes=2)
    semantic_maps = []
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                images = batch["image"].to(device)
                true_masks = batch["mask"].numpy()

                preds = model(images)
                iou_metric.update(preds.detach().cpu().numpy(), true_masks)

                class_labels = preds.argmax(dim=1)
                class_labels = class_labels.detach().cpu().numpy()
                final_predictions.append(class_labels.astype("uint8"))
                semantic_maps.append(class_labels.squeeze())

    except Exception as te:
        print(f"An exception occurred during inference: {te}")
    mean_iou = iou_metric.mean_iou()
    print(f"Mean IoU for the test dataset: {mean_iou}")
    final_predictions = np.concatenate(final_predictions, axis=0)
    store_semantic_maps(df_test, n_levels, semantic_maps)
    print(f"Semantic maps for next level are stored")
    return final_predictions
