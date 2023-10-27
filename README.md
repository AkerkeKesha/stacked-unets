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

## Repository Structure
```commandline
project-root/
|-- dataset/  # Data for training and testing
|-- notebooks/ # Jupyter notebooks for EDA, analysis, visualizations, etc
|-- output/ # Output files, such as predicted masks, plots, etc
|-- src/ # Source code for train, predict, analyze, etc
|-- tests/ # py tests
|-- .gitignore # untracked files to ignore
|-- config.py # Configuration file for dataset name, lr and epochs, etc
|-- gcolab_etci_flood_semantic_segmentation.ipynb # Example notebook for etci_flood in golab
|-- gcolab_spacenet6_semantic_segmentation.ipynb 
|-- README.md
|-- requirements.txt # Python packages
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

