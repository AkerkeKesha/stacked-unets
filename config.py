import os

environment = "colab"
# environment = "local"
if environment == "local":
    project_root = os.path.dirname(os.path.abspath(__file__))
elif environment == "colab":
    project_root = "/content/stacked-unets"
else:
    project_root = "/kaggle"

output_dir = os.path.join(project_root, 'output')
dataset = "etci"
# dataset = "sn6"
if dataset == "etci":
    dataset_name = "data-etci-flood"
    train_dir = os.path.join(project_root, "dataset", dataset_name, "train")
elif dataset == "sn6":
    dataset_name = "data-spacenet6"
    train_dir = os.path.join(project_root, "dataset", dataset_name, "train", "AOI_11_Rotterdam")
    mask_train_dir = os.path.join(train_dir, 'masks')
    sn6_summary_datapath = os.path.join(train_dir, "SummaryData", "SN6_Train_AOI_11_Rotterdam_Buildings.csv")

# hyperparameters
num_workers = 1 if environment == "local" else 2
batch_size = 2 if environment == "local" else 48

# training related
learning_rate = 1e-3
num_epochs = 2 if environment == "local" else 20


if dataset == "etci":
    SAMPLE_INDICES = [
        2078,
        2047,
        2496,
        2238,
        1672,
        1774,
        1579,
        1058,
        1810,
        625
    ]
else:
    SAMPLE_INDICES = []
