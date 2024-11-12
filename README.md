# Effective Non-Random Extreme Learning Machine

This repository contains the Python code to reproduce the results of the paper **"Effective Non-Random Extreme Learning Machine"**. Follow the instructions below to set up and run the code.

## Table of Contents

- [Requirements](#requirements)
- [Preparing the Datasets](#preparing-the-datasets)
- [Generating Synthetic Datasets](#generating-synthetic-datasets)
- [Running the Experiments](#running-the-experiments)

## Requirements

To ensure compatibility and proper functionality, use **Python 3.10**. Install the required Python packages as listed below:

- **`numpy`**
- **`scipy`**
- **`matplotlib`**
- **`scikit-learn`**

Install the packages using the following command:

```bash
pip install numpy scipy matplotlib scikit-learn
```
## Preparing the Datasets

Before running the code, download the real datasets and place them in a folder named `datasets`. If this folder does not exist, create it manually.

### Dataset Links

Download the datasets from the following links and organize them accordingly:

- **Abalone** - [Download Link](https://archive.ics.uci.edu/dataset/1/abalone)
- **Auto MPG** - [Download Link](https://archive.ics.uci.edu/dataset/9/auto+mpg)
- **California Housing** - [Download Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
- **Delta Ailerons** - [Download Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/delta_ailerons.html)
- **LA Ozone** - [Download Link](https://hastie.su.domains/ElemStatLearn/datasets/LAozone.data) (Save as `datasets/LAozone/LAozone.data.txt`)
- **Machine CPU** - [Download Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/machine.html)
- **Prostate Cancer** - [Download Link](https://hastie.su.domains/ElemStatLearn/datasets/prostate.data) (Save as `datasets/prostate/prostate.data.txt`)
- **Servo** - [Download Link](https://archive.ics.uci.edu/dataset/87/servo)

## Generating Synthetic Datasets

Before running any experiments, it is recommended to execute `datagenerator.py`. This script will generate the synthetic datasets used in the experiments.

Run the following command:

```bash
python datagenerator.py
```

## Running the Experiments

The repository includes two Jupyter notebooks:

1. **`experiments.ipynb`** – This notebook runs the experiments and produces `.csv` files containing the results.
2. **`presentation.ipynb`** – This notebook reads the results from `experiments.ipynb` and generates tables and plots for analysis.

### Generating Zoomed Plots

To create a plot with zoom that displays training and test errors for all models on real datasets, run `dynamicplot.py`:

```bash
python dynamicplot.py
```

This interactive plot allows you to zoom in on a specified x-axis range, which is useful for detailed analysis and was used to create Figure 4 of the paper. If zooming is not required, you can produce the plot directly using `presentation.ipynb`.
