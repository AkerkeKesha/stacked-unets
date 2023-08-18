import cv2
import torch.optim
import torch.nn as nn
import config
from tqdm.notebook import tqdm
from model import UNet
from utils import store_semantic_maps
from evaluate import IntersectionOverUnion


def train(num_epochs, train_loader, val_loader, df_train, df_val, level=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=5)if config.dataset == "sn6" else UNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        last_epoch = epoch
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

        train_ious.append(mean_iou)
        iou_metric.reset()
        training_loss = training_loss / len(train_loader)
        train_losses.append(training_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: [{epoch + 1} / {num_epochs}]")
            print(f"Train mean IoU = {mean_iou:.4f}")
            print(f"Train mean loss = {training_loss:.4f}")

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
                val_ious.append(mean_iou)
                iou_metric.reset()
                val_loss = val_loss / len(val_loader)
                val_losses.append(val_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Val mean IoU = {mean_iou:.4f}")
                    print(f"Val mean loss = {val_loss:.4f}")

                # Early stopping logic only at level 0
                if level == 0:
                    if best_val_loss - val_loss > config.early_stop_threshold:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= config.patience:
                            print(f"Early stopping triggered {last_epoch + 1}")
                            break

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
                for img in image:
                    img = img.unsqueeze(0)  # add batch dimension because model expects it
                    semantic_map = model(img).argmax(dim=1).cpu().numpy()
                    target_dimensions = (img.shape[3], img.shape[2])  # (width, height)
                    semantic_map = cv2.resize(semantic_map[0], target_dimensions)
                    semantic_map = semantic_map.astype("uint8")
                    semantic_maps.append(semantic_map)

            if split == "train":
                df_train = store_semantic_maps(df, level, semantic_maps)
            elif split == "val":
                df_val = store_semantic_maps(df, level, semantic_maps)

    torch.save(model.state_dict(), f"{config.output_dir}/level_{level}_unet_{config.dataset}.pt")
    return train_losses, val_losses, train_ious, val_ious, df_train, df_val




