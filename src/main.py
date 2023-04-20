import time
import math
import numpy as np
import matplotlib.pyplot as plt
from train import train
from predict import predict
import config
from utils import (
    visualize_prediction, get_etci_df, cleanup_etci_data, get_sn6_df
)


def start_basic_unet():
    start = time.time()
    train_losses, val_losses, train_iou, val_iou = train(config.num_epochs)
    print(f"{time.time() - start} seconds to train")
    np.save(f'{config.output_dir}/train_losses_{config.dataset}.npy', train_losses)
    np.save(f'{config.output_dir}/val_losses_{config.dataset}.npy', val_losses)
    print(f"Done saving average losses")
    np.save(f'{config.output_dir}/train_iou_{config.dataset}.npy', train_iou)
    np.save(f'{config.output_dir}/val_iou_{config.dataset}.npy', val_iou)
    print(f"Done saving evaluation metrics")

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

    if config.dataset == "sn6":
        test_df = get_sn6_df(split="test")
    else:
        test_df = get_etci_df(config.test_dir, split="test")
        test_df = cleanup_etci_data(test_df)
    final_predictions = predict(test_df, config.dataset)

    np.save(f'{config.output_dir}/predictions_{config.dataset}.npy', final_predictions, fix_imports=True, allow_pickle=False)

    n_batches = math.ceil(len(final_predictions) / config.batch_size)
    for batch in range(n_batches):
        start = batch * config.batch_size
        end = min((batch + 1) * config.batch_size, len(final_predictions))

        for index in range(start, end):
            # TODO: visualize spacenet6
            visualize_prediction(test_df.iloc[index], final_predictions[index],
                                 output_dir=f'{config.output_dir}/{config.dataset}_labels', figure_size=(10, 6))

        print(f"Finished plotting batch {batch + 1}/{n_batches}")
    print(f"All predictions finished plotting")
