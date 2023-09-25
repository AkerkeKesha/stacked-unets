import os

environment = os.getenv("ENVIRONMENT", "local")
if environment == "local":
    project_root = os.path.dirname(os.path.abspath(__file__))
elif environment == "colab":
    project_root = "/content/stacked-unets"
else:
    project_root = "/kaggle"

output_dir = os.path.join(project_root, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset = os.getenv("DATASET", "etci")
labels_dir = f"{output_dir}/{dataset}_labels"
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

if dataset == "etci":
    dataset_name = "data-etci-flood"
    train_dir = os.path.join(project_root, "dataset", dataset_name, "train")
elif dataset == "sn6":
    dataset_name = "data-spacenet6"
    train_dir = os.path.join(project_root, "dataset", dataset_name, "train", "AOI_11_Rotterdam")
    mask_train_dir = os.path.join(train_dir, 'masks')
    sn6_summary_datapath = os.path.join(train_dir, "SummaryData", "SN6_Train_AOI_11_Rotterdam_Buildings.csv")

# if we want to stop early at level 0, then stop = yes
stop = os.getenv("STOP", "no")

# hyperparameters
num_workers = 1 if environment == "local" else 2
batch_size = 2 if environment == "local" else 48

# training related
learning_rate = 1e-3
num_epochs = 2 if environment == "local" else 20
