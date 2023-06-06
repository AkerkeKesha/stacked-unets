import torch.optim
import torch.nn as nn
import config
from tqdm.notebook import tqdm
from model import StackedUNet, basic_unet
from utils import store_semantic_maps
from evaluate import IntersectionOverUnion


def train(num_epochs, train_loader, val_loader, df_train, df_val,  n_levels=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StackedUNet(n_levels=n_levels, base_model=basic_unet)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    for level in range(n_levels + 1):
        for epoch in range(num_epochs):
            print(f"Epoch: [{epoch + 1} / {num_epochs}]")
            model.train()

            training_loss = 0
            iou_metric = IntersectionOverUnion(num_classes=2)
            try:
                for batch in tqdm(train_loader):
                    image = batch["image"].to(device)
                    mask = batch["mask"].to(device)
                    pred = model(image)
                    loss = criterion(pred, mask)
                    optimizer.zero_grad()
                    training_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())
            except Exception as e:
                print(f"An exception occurred during training: {e}")
                continue

            mean_iou = iou_metric.mean_iou()
            print(f"Train mean IoU = {mean_iou:.4f}")
            train_ious.append(mean_iou)
            iou_metric.reset()
            training_loss = training_loss / len(train_loader)
            print(f"Train mean loss = {training_loss:.4f}")
            train_losses.append(training_loss)

            model.eval()
            iou_metric = IntersectionOverUnion(num_classes=2)
            val_loss = 0
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
                    val_ious.append(mean_iou)
                    iou_metric.reset()
                    val_loss = val_loss / len(val_loader)
                    print(f"Val mean loss = {val_loss:.4f}")
                    val_losses.append(val_loss)
            except Exception as ve:
                print(f"An exception occurred during validation: {ve}")
                continue
        for split, df, loader in [("train", df_train, train_loader), ("val", df_val, val_loader)]:
            model.eval()
            print(f"Generating semantic maps for {split} dataset...")
            with torch.no_grad():
                semantic_maps = []
                for batch in tqdm(loader):
                    image = batch["image"].to(device)
                    semantic_map = model(image).argmax(dim=1).cpu().numpy()
                    semantic_maps.extend(semantic_map)

                store_semantic_maps(df, n_levels, semantic_maps)

    torch.save(model.state_dict(), f"{config.output_dir}/level_{n_levels}_unet_{config.dataset}.pt")
    return train_losses, val_losses, train_ious, val_ious




