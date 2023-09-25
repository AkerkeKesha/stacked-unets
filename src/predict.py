import torch
from torch.nn.functional import softmax
import numpy as np
from tqdm import tqdm
from model import UNet
from evaluate import IntersectionOverUnion
from utils import store_semantic_maps
import config


def predict(test_loader, df_test, level=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(f"{config.output_dir}/level{level}_unet_{config.dataset}.pt"))
    model.to(device)
    model.eval()

    final_predictions, true_labels = [], []
    iou_metric = IntersectionOverUnion(num_classes=2)
    semantic_maps = []
    entropy_values = []
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                true_masks = batch["mask"].numpy()
                for image, true_mask in zip(batch["image"], true_masks):
                    image = image.unsqueeze(0).to(device)  # add batch dimension because model expects it
                    pred = model(image)

                    softmax_probs = softmax(pred, dim=1)
                    entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-9),
                                         dim=1)  # Sum over the classes, avoid log(0) by adding a small constant
                    entropy = entropy.detach().cpu().numpy()
                    entropy_values.append(np.mean(entropy))

                    iou_metric.update(pred.detach().cpu().numpy(), true_mask)

                    class_label = pred.argmax(dim=1)
                    class_label = class_label.detach().cpu().numpy()
                    final_predictions.append(class_label.astype("uint8"))
                    semantic_maps.append(class_label.squeeze())

    except Exception as te:
        print(f"An exception occurred during inference: {te}")
    mean_iou = iou_metric.mean_iou()
    print(f"Mean IoU for the test dataset: {mean_iou}")
    overall_avg_entropy = np.mean(entropy_values)
    print(f"Overall average entropy for the entire test set: {overall_avg_entropy:.4f}")
    final_predictions = np.concatenate(final_predictions, axis=0)
    df_test = store_semantic_maps(df_test, level, semantic_maps)
    return final_predictions, df_test, mean_iou
