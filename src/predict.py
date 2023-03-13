import torch
from torch.utils.data import DataLoader
from dataset import ETCIDataset
from model import create_single_unet
import config
from tqdm.notebook import tqdm
from utils import create_df


def get_test_df():
    return create_df(config.test_dir, split="test")


def predict():
    final_predictions = []
    test_df = get_test_df()
    test_dataset = ETCIDataset(test_df, split="test", transform=None)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_single_unet()
    model.load_state_dict(torch.load("output/single_unet.pt"))
    model.to(device)

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            image = batch["image"].to(device)
            pred = model(image)
            class_label = pred.argmax(dim=1)
            class_label = class_label.detach().cpu().numpy()
            final_predictions.append(class_label)

    return final_predictions
