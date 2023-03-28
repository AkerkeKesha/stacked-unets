import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from dataset import ETCIDataset
from model import create_single_unet
import config
from utils import get_etci_df, cleanup_etci_data


def predict(test_df):
    final_predictions = []

    test_dataset = ETCIDataset(test_df, split="test", transform=None)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_single_unet()
    model.load_state_dict(torch.load(f"{config.output_dir}/single_unet.pt"))
    model.to(device)
    model.double()
    model.eval()
    try:
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image = batch["image"].to(device)
                pred = model(image)
                class_label = pred.argmax(dim=1)
                class_label = class_label.detach().cpu().numpy()
                final_predictions.append(class_label.astype("uint8"))
    except Exception as te:
        print(f"An exception occurred during inference: {te}")

    # produce a single array of prediction from all batches of test data
    final_predictions = np.concatenate(final_predictions, axis=0)
    return final_predictions
