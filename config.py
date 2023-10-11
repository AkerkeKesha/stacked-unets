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
num_channels = 3 if dataset == "etci" else 5
labels_dir = f"{output_dir}/{dataset}_labels"
mask_train_dir = None
sn6_summary_datapath = None

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
output_type = os.getenv("OUTPUT_TYPE", "semantic_map")

metrics = ["train_loss", "val_loss", "train_iou", "val_iou", "test_iou", "timing", "entropy"]

# hyperparameters
num_workers = 1 if environment == "local" else 2
batch_size = 16 if dataset == "sn6" else 48

# training related
learning_rate = 1e-2 if dataset == "sn6" else 1e-3
num_epochs = 20

# input feature mean/std for normalization
mean_vv, mean_vh = 0.56183946, 0.7116502
std_vv, std_vh = 0.18815923, 0.20106803

mean_sar_image, std_sar_image = 0.0, 0.0  # change

if output_type == "semantic_map":
    if dataset == "sn6":
        mean_sem_map = 1.0
        std_sem_map = 0.0
    elif dataset == "etci":
        mean_semantic_map = 0.03615002
        std_semantic_map = 0.05179913
elif output_type == "softmax_prob":
    if dataset == "sn6":
        mean_softmax_prob = 0.0241  # change
        std_softmax_prob = 0.1044   # change
    elif dataset == "etci":
        mean_softmax_prob = 0.0462399
        std_softmax_prob = 0.13435808
