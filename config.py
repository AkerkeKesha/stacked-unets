import os
import argparse


# parser = argparse.ArgumentParser(description='Run basic UNet')
# parser.add_argument('--colab', dest='is_colab', type=bool, help='Flag to indicate if running on Colab or not')
# parser.add_argument('--dataset', dest='dataset_name', type=str, default='etci', help='Name of the dataset (etci or sn6)')
# args = parser.parse_args()
# is_colab = args.is_colab
# dataset_name = arg.dataset_name
dataset = "etci"
# dataset_name = "sn6"
# is_colab = False
is_colab = True
if not is_colab:
    project_root = os.path.dirname(os.path.abspath(__file__))
else:
    project_root = '/content/stacked-unets'

output_dir = os.path.join(project_root, 'output')

if dataset == 'etci':
    dataset_name = 'data-etci-flood'
    train_dir = os.path.join(project_root, 'dataset', dataset_name, 'train')
    test_dir = os.path.join(project_root, 'dataset', dataset_name, 'test_internal')
elif dataset == 'sn6':
    dataset_name = 'data-spacenet6'
    train_dir = os.path.join(project_root, 'dataset', dataset_name, 'train', 'AOI_11_Rotterdam')
    test_dir = os.path.join(project_root, 'dataset', dataset_name, 'test_public', 'AOI_11_Rotterdam')
    mask_train_dir = os.path.join(train_dir, 'masks')
    sn6_summary_datapath = os.path.join(train_dir, 'SummaryData', 'SN6_Train_AOI_11_Rotterdam_Buildings.csv')

# hyperparameters
num_workers = 2
batch_size = 96

# training related
learning_rate = 1e-3
num_epochs = 5
