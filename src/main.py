import time
from train import train
from predict import predict
import numpy as np
import config
from utils import visualize_prediction, get_etci_df
import matplotlib.pyplot as plt


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
    np.save(f'{config.output_dir}/predictions.npy',  fix_imports=True, allow_pickle=False)
    # TODO: a couple of random indices or all
    index = -100
    test_df = get_etci_df(config.test_dir, split="test")
    visualize_prediction(test_df.iloc[index], final_predictions[index], figure_size=(17, 10))

