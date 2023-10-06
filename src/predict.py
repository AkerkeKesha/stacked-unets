import torch
from torch.nn.functional import softmax
import numpy as np
from tqdm import tqdm
from model import UNet
from evaluate import IntersectionOverUnion
from utils import store_semantic_maps, store_softmax_probs
import config


def predict(test_loader, df_test, level=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=5) if config.dataset == "sn6" else UNet()
    model.load_state_dict(torch.load(f"{config.output_dir}/level{level}_unet_{config.dataset}.pt"))
    model.to(device)
    model.eval()

    sum_mean = torch.zeros([3], dtype=torch.float32).to(device)
    sum_std = torch.zeros([3], dtype=torch.float32).to(device)

    final_predictions, true_labels = [], []
    iou_metric = IntersectionOverUnion(num_classes=2)
    semantic_maps, entropy_values = [], []
    collected_softmax_probs = []
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                true_masks = batch["mask"].numpy()
                for image, true_mask in zip(batch["image"], true_masks):
                    image = image.unsqueeze(0).to(device)  # add batch dimension because model expects it

                    batch_mean = torch.mean(image, dim=[0, 2, 3])
                    batch_std = torch.std(image, dim=[0, 2, 3])
                    sum_mean += batch_mean
                    sum_std += batch_std

                    pred = model(image)

                    softmax_output = softmax(pred, dim=1)
                    entropy = -torch.sum(softmax_output * torch.log(softmax_output + 1e-9), dim=1)
                    entropy = entropy.detach().cpu().numpy()
                    entropy_values.append(np.mean(entropy))
                    probs = softmax_output[:, 1, :, :]  # Probability of the second class
                    probs = probs.cpu().numpy()
                    collected_softmax_probs.append(probs[0])

                    iou_metric.update(pred.detach().cpu().numpy(), true_mask)

                    class_label = pred.argmax(dim=1)
                    class_label = class_label.detach().cpu().numpy()
                    final_predictions.append(class_label.astype("uint8"))
                    semantic_maps.append(class_label.squeeze())

            avg_mean = sum_mean / len(test_loader)
            avg_std = sum_std / len(test_loader)

            print(f"Input Feature Mean in Test: {avg_mean.cpu().numpy()}")
            print(f"Input Feature Std in Test: {avg_std.cpu().numpy()}")

    except Exception as te:
        print(f"An exception occurred during inference: {te}")
    mean_iou = iou_metric.mean_iou()
    print(f"Mean IoU for the test dataset: {mean_iou}")
    overall_avg_entropy = np.mean(entropy_values)
    print(f"Overall average entropy for the entire test set: {overall_avg_entropy:.4f}")
    final_predictions = np.concatenate(final_predictions, axis=0)
    if config.output_type == "semantic_map":
        df_test = store_semantic_maps(df_test, level, semantic_maps)
    elif config.output_type == "softmax_prob":
        df_test = store_softmax_probs(df_test, level, collected_softmax_probs)
    else:
        raise ValueError("Invalid output type")
    return final_predictions, df_test, mean_iou, overall_avg_entropy
