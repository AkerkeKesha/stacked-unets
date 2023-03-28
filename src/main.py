import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from train import train
from predict import predict
import config
from utils import visualize_prediction, get_etci_df, cleanup_etci_data


def plot_all_predictions(test_df, final_predictions, output_dir):
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        prediction = final_predictions[i]
        visualize_prediction(row, prediction, figure_size=(17, 10))
        plt.savefig(os.path.join(output_dir, f"plot_{i}.png"))


def start_basic_unet():
    start = time.time()
    train_losses, val_losses = train(config.num_epochs)
    print(f"{time.time() - start} seconds to train")
    np.save(f'{config.output_dir}/train_losses.npy', train_losses)
    np.save(f'{config.output_dir}/val_losses.npy', val_losses)
    print(f"Done saving average losses")

    train_losses = np.load(f'{config.output_dir}/train_losses.npy')
    val_losses = np.load(f'{config.output_dir}/val_losses.npy')
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss values")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{config.output_dir}/loss_plot.png', bbox='tight')
    plt.show()

    final_predictions = predict()
    np.save(f'{config.output_dir}/predictions.npy', final_predictions, fix_imports=True, allow_pickle=False)
    test_df = get_etci_df(config.test_dir, split="test")
    test_df = cleanup_etci_data(test_df)

    output_dir = f"{config.target_dir}"
    plot_all_predictions(test_df, final_predictions, output_dir)
    print(f"Prediction plots saved")

