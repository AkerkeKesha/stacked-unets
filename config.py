import os

project_root = os.path.dirname(os.path.abspath(__file__))
# dataset related
etci_dataset = os.path.join(project_root, 'dataset', 'data-etci-flood')
train_dir = os.path.join(etci_dataset, "train")
test_dir = os.path.join(etci_dataset, "test_internal")

# hyperparameters
num_workers = 8
batch_size = 96

# training related
learning_rate = 1e-3
num_epochs = 10


