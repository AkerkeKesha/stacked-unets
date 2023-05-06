import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import config
from train import train
from predict import predict
from dataloader import get_loader
from utils import plot_single_prediction


def start_basic_unet():
    start = time.time()
    train_loader, val_loader, test_loader = get_loader(config.dataset)
    train_losses, val_losses, train_iou, val_iou = train(config.num_epochs, train_loader, val_loader)
    print(f"{time.time() - start} seconds to train")
    np.save(f'{config.output_dir}/train_losses_{config.dataset}.npy', train_losses)
    np.save(f'{config.output_dir}/val_losses_{config.dataset}.npy', val_losses)
    print(f"Done saving average losses")
    np.save(f'{config.output_dir}/train_iou_{config.dataset}.npy', train_iou)
    np.save(f'{config.output_dir}/val_iou_{config.dataset}.npy', val_iou)
    print(f"Done saving evaluation metrics on train/val")

    train_losses = np.load(f'{config.output_dir}/train_losses_{config.dataset}.npy')
    val_losses = np.load(f'{config.output_dir}/val_losses_{config.dataset}.npy')
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss values")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{config.output_dir}/loss_plot_{config.dataset}.png', bbox_inches='tight')
    plt.show()

    train_iou = np.load(f'{config.output_dir}/train_iou_{config.dataset}.npy')
    val_iou = np.load(f'{config.output_dir}/val_iou_{config.dataset}.npy')
    epochs = range(1, len(train_iou) + 1)
    plt.plot(epochs, train_iou, label="Training Mean IoU")
    plt.plot(epochs, val_iou, label="Validation Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.legend()
    plt.title("Mean IoU over Epochs")
    plt.savefig(f'{config.output_dir}/iou_plot_{config.dataset}.png', bbox_inches='tight')
    plt.show()

    final_predictions = predict(test_loader)
    np.save(f'{config.output_dir}/predictions_{config.dataset}.npy', final_predictions, fix_imports=True, allow_pickle=False)
    n_batches = math.ceil(len(final_predictions) / config.batch_size)
    for batch in range(n_batches):
        start = batch * config.batch_size
        end = min((batch + 1) * config.batch_size, len(final_predictions))
        for index in range(start, end):
            original_index = index
            df_row = test_loader.dataset.dataset.iloc[original_index]
            vv_image_path = df_row["vv_image_path"]
            image_id = os.path.basename(vv_image_path).split('.')[0]
            plot_single_prediction(image_id, final_predictions[index],
                                   f"{config.output_dir}/{config.dataset_name}_labels)")

        print(f"Finished plotting batch {batch + 1}/{n_batches}")
    print(f"All predictions finished plotting")
