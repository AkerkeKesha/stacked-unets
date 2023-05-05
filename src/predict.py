import torch
import numpy as np
from tqdm.notebook import tqdm
from model import create_single_unet
from evaluate import IntersectionOverUnion
import config


def predict(test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_single_unet()
    model.load_state_dict(torch.load(f"{config.output_dir}/single_unet_{config.dataset}.pt"))
    model.to(device)
    model.eval()

    final_predictions, true_labels = [], []
    iou_metric = IntersectionOverUnion(num_classes=2)
    try:
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image = batch["image"].to(device)
                true_mask = batch["mask"].numpy()
                print(f"[predict] Image shape: {image.shape}, True mask shape: {true_mask.shape}")
                true_labels.append(true_mask)

                pred = model(image)
                iou_metric.update(pred.detach().cpu().numpy(), true_mask)  # Pass the pred tensor directly

                class_label = pred.argmax(dim=1)
                class_label = class_label.detach().cpu().numpy()
                final_predictions.append(class_label.astype("uint8"))

    except Exception as te:
        print(f"An exception occurred during inference: {te}")

    mean_iou = iou_metric.mean_iou()
    print(f"Mean IoU for the test dataset: {mean_iou}")
    final_predictions = np.concatenate(final_predictions, axis=0)  # a single array of prediction from all batches
    return final_predictions
