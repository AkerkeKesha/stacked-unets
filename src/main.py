import time
from train import train
from predict import predict, get_test_df
import numpy as np
import config
from utils import visualize_pred
import matplotlib.pyplot as plt


if __name__ == '__main__':
    start = time.time()
    train_losses, val_losses = train(config.num_epochs)
    print(f"{time.time() - start} seconds to train")
    np.save('train_losses.npy', train_losses)
    np.save('val_losses.npy', val_losses)
    print(f"Done saving average losses")
    train_losses = np.load('train_losses.npy')
    val_losses = np.load('val_losses.npy')
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss values")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    final_predictions = predict()
    index = -100
    test_df = get_test_df()
    visualize_pred(test_df.iloc[index], final_predictions[index], figsize=(17, 10))
