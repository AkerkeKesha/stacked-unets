import os

is_colab = True

if not is_colab:
    project_root = os.path.dirname(os.path.abspath(__file__))
    # dataset related
    etci_dataset = os.path.join(project_root, 'dataset', 'data-etci-flood')
    train_dir = os.path.join(etci_dataset, "train")
    test_dir = os.path.join(etci_dataset, "test_internal")
    output_dir = os.path.join(project_root, 'output')
else:
    # dataset related
    etci_dataset = '/content/ETCI_2021_Competition_Dataset/'
    train_dir = os.path.join(etci_dataset, "train")
    test_dir = os.path.join(etci_dataset, "val")
    output_dir = os.path.join('content', 'output')


# hyperparameters
num_workers = 2
batch_size = 96

# training related
learning_rate = 1e-3
num_epochs = 5




