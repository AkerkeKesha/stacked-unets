import cv2
import torch.optim
import torch.nn as nn
import config
from tqdm.notebook import tqdm
from src.model import UNet
from src.utils import store_semantic_maps, store_softmax_probs
from src.evaluate import IntersectionOverUnion


def train(train_loader, val_loader, df_train, df_val, level, run_key):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=config.num_channels)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    num_epochs = 5 if (level == 0 and config.stop == "yes") else config.num_epochs

    for epoch in range(num_epochs):
        model.train()
        training_loss = 0
        sum_mean = torch.zeros([config.num_channels], dtype=torch.float32).to(device)
        sum_std = torch.zeros([config.num_channels], dtype=torch.float32).to(device)
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

                batch_mean = torch.mean(image, dim=[0, 2, 3])
                batch_std = torch.std(image, dim=[0, 2, 3])
                sum_mean += batch_mean
                sum_std += batch_std

                iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())
        except Exception as e:
            print(f"An exception occurred during training: {e}")
            continue

        mean_iou = iou_metric.mean_iou()

        train_ious.append(mean_iou)
        iou_metric.reset()
        training_loss = training_loss / len(train_loader)
        train_losses.append(training_loss)

        avg_mean = sum_mean / len(train_loader)
        avg_std = sum_std / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: [{epoch + 1} / {num_epochs}]")
            print(f"Train mean IoU = {mean_iou:.4f}")
            print(f"Train mean loss = {training_loss:.4f}")
            print(f"Input Feature Mean in train: {avg_mean.cpu().numpy()}")
            print(f"Input Feature Std in train: {avg_std.cpu().numpy()}")

        model.eval()
        iou_metric = IntersectionOverUnion(num_classes=2)
        val_loss = 0
        sum_mean = torch.zeros([config.num_channels], dtype=torch.float32).to(device)
        sum_std = torch.zeros([config.num_channels], dtype=torch.float32).to(device)
        try:
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    image = batch["image"].to(device)
                    mask = batch["mask"].to(device)
                    pred = model(image)
                    loss = criterion(pred, mask)
                    val_loss += loss.item()

                    batch_mean = torch.mean(image, dim=[0, 2, 3])
                    batch_std = torch.std(image, dim=[0, 2, 3])
                    sum_mean += batch_mean
                    sum_std += batch_std

                    iou_metric.update(pred.detach().cpu().numpy(), mask.cpu().numpy())

                mean_iou = iou_metric.mean_iou()
                val_ious.append(mean_iou)
                iou_metric.reset()
                val_loss = val_loss / len(val_loader)
                val_losses.append(val_loss)

                avg_mean = sum_mean / len(val_loader)
                avg_std = sum_std / len(val_loader)

                if (epoch + 1) % 10 == 0:
                    print(f"Val mean IoU = {mean_iou:.4f}")
                    print(f"Val mean loss = {val_loss:.4f}")
                    print(f"Input Feature Mean in val: {avg_mean.cpu().numpy()}")
                    print(f"Input Feature Std in val: {avg_std.cpu().numpy()}")

        except Exception as ve:
            print(f"An exception occurred during validation: {ve}")
            break

    for split, df, loader in [("train", df_train, train_loader), ("val", df_val, val_loader)]:
        model.eval()
        with torch.no_grad():
            semantic_maps = []
            softmax_probs = []
            for batch in tqdm(loader):
                image = batch["image"].to(device)
                for img in image:
                    img = img.unsqueeze(0)  # add batch dimension because model expects it
                    semantic_map = model(img).argmax(dim=1).cpu().numpy()
                    target_dimensions = (img.shape[3], img.shape[2])  # (width, height)
                    semantic_map = cv2.resize(semantic_map[0], target_dimensions)
                    semantic_map = semantic_map.astype("uint8")
                    semantic_maps.append(semantic_map)

                    probs = torch.softmax(model(img), dim=1)[:, 1, :, :]  # Probability of the second class
                    probs = probs.cpu().numpy()
                    softmax_probs.append(probs[0])

            if split == "train":
                if config.output_type == "semantic_map":
                    df_train = store_semantic_maps(df, level, semantic_maps)
                elif config.output_type == "softmax_prob":
                    df_train = store_softmax_probs(df, level, softmax_probs)
                else:
                    raise ValueError("Invalid output type")
            elif split == "val":
                if config.output_type == "semantic_map":
                    df_val = store_semantic_maps(df, level, semantic_maps)
                elif config.output_type == "softmax_prob":
                    df_val = store_softmax_probs(df, level, softmax_probs)
                else:
                    raise ValueError("Invalid output type")

    torch.save(model.state_dict(), f"{config.output_dir}/{run_key}_level{level}_unet_{config.dataset}.pt")
    return train_losses, val_losses, train_ious, val_ious, df_train, df_val




