import torch.optim
import torch.nn as nn
import config
from tqdm.notebook import tqdm
from model import create_single_unet
from utils import get_logging
from dataloader import get_loader
from evaluate import IntersectionOverUnion


def train(num_epochs):
    logging = get_logging()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_single_unet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    iou_metric = IntersectionOverUnion(num_classes=2)
    train_loader, val_loader = get_loader()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch: [{epoch + 1} / {num_epochs}]")
        model.train()
        for batch in tqdm(train_loader):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())
        mean_iou = iou_metric.mean_iou()
        print(f"Train mean IoU = {mean_iou:.4f}")
        iou_metric.reset()
    model.eval()
    iou_metric = IntersectionOverUnion(num_classes=2)
    with torch.no_grad():
        for batch in tqdm(val_loader):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            loss = criterion(pred, mask)
            val_losses.append(loss.item())
            iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())
        mean_iou = iou_metric.mean_iou()
        print(f"Val mean IoU = {mean_iou:.4f}")
        iou_metric.reset()

    torch.save(model.state_dict(), "output/single_unet.pt")
    return train_losses, val_losses
