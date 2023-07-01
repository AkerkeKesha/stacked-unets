import os

environment = "colab"
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
    SAMPLE_IMAGES = [
        "bangladesh_20170606t115613_x-33_y-29",
        "bangladesh_20170314t115609_x-11_y-20",
        "bangladesh_20170314t115609_x-2_y-39",
        "nebraska_20171210t002119_x-13_y-28",
        "nebraska_20170731t002118_x-9_y-26",
        "nebraska_20170731t002118_x-10_y-14",
        "northal_20191227t234659_x-9_y-9",    # this prediction looks bad
        "northal_20190407t234651_x-19_y-18",
        "northal_20190712t234656_x-5_y-25",
    ]
else:
    SAMPLE_INDICES = []
