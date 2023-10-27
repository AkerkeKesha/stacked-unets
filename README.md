# stacked-unets

### Getting Started

To view the code, please follow these steps:

Clone the repository by running the following command in your terminal:
```commandline
git clone https://github.com/AkerkeKesha/stacked-unets.git
```
Navigate to the cloned repository directory:

```commandline
cd stacked-unets
```

Create a new virtual environment using the following command:

```bash
python3 -m venv env
```
This will create a new folder named env in your project directory, which will contain a Python environment with its own installed packages and interpreter.

Activate the virtual environment by running the following command:

```bash
source env/bin/activate
````
This will activate the virtual environment and change the terminal prompt to indicate that you are now working in the virtual environment.

Once the virtual environment is activated, install the required packages using the `requirements.txt` file with the following command:
```commandline
pip install -r requirements.txt
```
This will install all the packages listed in the requirements.txt file in the virtual environment.
You can now edit the code as needed while working within the virtual environment.
When you're done working on the project, you can deactivate the virtual environment by running the following command:
```commandline
deactivate
```
This will return the terminal prompt to its default state, indicating that you are no longer working in the virtual environment.
### Data Preparation

This project uses the ETCI 2021 Flood Detection dataset. 

#### Steps to Download the ETCI Dataset
1. **Navigate to the Kaggle Website**: [ETCI 2021 Flood Detection Data Card](https://www.kaggle.com/datasets/aninda/etci-2021-competition-on-flood-detection/) to access the dataset.
  
2. **Register/Log in**: Follow the prompts to register or log into Kaggle

3. **Download**: After successful login, navigate to the "Download" section and download the required dataset files.

4. **Unzip the Dataset**: Unzip the downloaded files into a folder.

####  Dataset Placement
After downloading and unzipping the dataset, move it into the `dataset/` folder in this repository.

```bash
# Move the data to the dataset/ directory
mv path/to/unzipped/data/* path/to/your/repo/dataset/
```

### Repository Structure
```commandline
project-root/
|-- dataset/  # data
|-- notebooks/ # notebooks for EDA, analysis, visualizations, etc
|-- output/ # output files, such as predicted masks, plots, etc
|-- src/ # source code for train, predict, analyze, etc
|-- tests/ # py tests
|-- .gitignore # untracked files to ignore
|-- config.py # configuration file for dataset name, lr and epochs, etc
|-- gcolab_etci_flood_semantic_segmentation.ipynb # Example notebook for etci_flood in golab
|-- gcolab_spacenet6_semantic_segmentation.ipynb 
|-- README.md
|-- requirements.txt # python packages
|-- set_env.sh # Environment variables to set dataset, etc
```

### Explore notebooks
Virtual environment can be installed as kernel in jupyter notebook
```commandline
python3 -m ipykernel install --user --name=env
```
Now, jupyter notebooks are viewed by:
```commandline
jupyter notebook
```
### Google Colab for Training
If you don't have the computational resources to train the model locally, 
you can use Google Colab for training. 

### Initial Setup

1. **Upload the Notebook**: Upload `gcolab_etci_flood_semantic_segmentation.ipynb` to Google Colab.

2. **Upload the Data**: 
    - Compress the `dataset/` directory to `dataset.zip`
    - Upload `dataset.zip` to Google Drive.

3. **Mount**: Mount the Google drive in order to copy data into Colab space

### Setting Environment Variables
Set up some environment variables before running the experiment. 
Create a code cell in your Colab notebook and paste the following:

```python
import os

os.environ['DATASET'] = 'etci'
os.environ['ENVIRONMENT'] = 'colab'  # Set environment to 'colab'
os.environ['STOP'] = 'yes'
os.environ['OUTPUT_TYPE'] = 'softmax_prob'
```
Alternatively, have set_env.sh uploaded, then run:

```commandline
source set_env.sh
```

### Running the Experiment
After env variables setting up, run the experiment using the run_experiments function.
Modify the parameters accordingly.
```python
run_experiments(runs=1, n_levels=2, max_data_points=1000)
```
`runs`: The number of times to run the experiment.
`n_levels`: The number of levels for stacking.
`max_data_points`: The number of data points to use (useful for quick testing).

