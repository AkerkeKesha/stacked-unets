import torch.optim
import torch.nn as nn
import config
from tqdm.notebook import tqdm
from model import create_single_unet
from dataloader import get_loader
from evaluate import IntersectionOverUnion


def train(num_epochs):
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
        training_loss = 0
        try:
            for batch in tqdm(train_loader):
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                pred = model(image)

                loss = criterion(pred, mask)
                optimizer.zero_grad()
                training_loss += loss.item()
                # TODO: dont learn on wrong parts of the images, e.f. reg = 0, is land not water
                loss.backward()
                optimizer.step()
                iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())
        except Exception as e:
            print(f"An exception occurred during training: {e}")
            continue

        mean_iou = iou_metric.mean_iou()
        print(f"Train mean IoU = {mean_iou:.4f}")
        iou_metric.reset()
        training_loss = training_loss / len(train_loader)
        print(f"Train mean loss = {training_loss:.4f}")
        train_losses.append(training_loss)
        model.eval()
        iou_metric = IntersectionOverUnion(num_classes=2)
        try:
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    image = batch["image"].to(device)
                    mask = batch["mask"].to(device)
                    pred = model(image)
                    loss = criterion(pred, mask)
                    val_loss += loss.item()
                    iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())

                mean_iou = iou_metric.mean_iou()
                print(f"Val mean IoU = {mean_iou:.4f}")
                iou_metric.reset()
                val_loss = val_loss / len(val_loader)
                print(f"Val mean loss = {val_loss:.4f}")
                val_losses.append(val_loss)
        except Exception as ve:
            print(f"An exception occurred during validation: {ve}")
            continue

    torch.save(model.state_dict(), "output/single_unet.pt")
    return train_losses, val_losses




