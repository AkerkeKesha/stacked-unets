import os

environment = ""
try:
    from google.colab import drive
    environment = "colab"
except ImportError:
    pass

try:
    import kaggle
    environment = "kaggle"
except ImportError:
    pass

if environment == "":
    environment = "local"

if environment == "local":
    project_root = os.path.dirname(os.path.abspath(__file__))
elif environment == "colab":
    project_root = "/content/stacked-unets"
else:
    project_root = "/kaggle"

output_dir = os.path.join(project_root, 'output')
# dataset = "etci"
dataset = "sn6"
if dataset == 'etci':
    dataset_name = 'data-etci-flood'
    train_dir = os.path.join(project_root, 'dataset', dataset_name, 'train')
elif dataset == 'sn6':
    dataset_name = 'data-spacenet6'
    train_dir = os.path.join(project_root, 'dataset', dataset_name, 'train', 'AOI_11_Rotterdam')
    mask_train_dir = os.path.join(train_dir, 'masks')
    sn6_summary_datapath = os.path.join(train_dir, 'SummaryData', 'SN6_Train_AOI_11_Rotterdam_Buildings.csv')

# hyperparameters
num_workers = 2
batch_size = 48

# training related
learning_rate = 1e-3
num_epochs = 20

