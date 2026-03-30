# M2 Coursework: Deep Learning

This repository contains the coursework code for the M2 Deep Learning module.

## Overview
The coursework involves implementing, training, and testing a Normalizing Flow model using PyTorch on the two moons dataset.

## Repository Structure
- `coursework.ipynb`: The main notebook containing all the modeling, training, testing, and evaluation code.
- `pyproject.toml`: The configuration file specifying project dependencies.
- `figs/`: Contains the generated PDF figures from the notebook execution.
- `checkpoints/`: Contains the saved model weights (`flow_full.pt`).
- `logs/`: Contains the training logs (`training_curves.json`).
- `results.json`: A generated JSON file containing the test evaluation metrics and the final written write-up.

## Usage
The notebook is designed to be fully reproducible on a CPU-only machine.

### Recommended setup (with local virtual environment)

```bash
python3 -m venv m2_env
source m2_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Register a notebook kernel

```bash
python -m ipykernel install --user --name m2-deep-learning-coursework --display-name "Python (m2-deep-learning-coursework)"
```

Then open `coursework.ipynb`, select `Python (m2-deep-learning-coursework)` as the kernel, and run the notebook from top to bottom.

The notebook will generate required output artifacts in `figs/`, `checkpoints/`, and `logs/`, plus `results.json`.

### Reproducibility

- The notebook pins execution to CPU (`CUDA_VISIBLE_DEVICES=-1`) and sets a global random seed.

## Use of Generative Tools

This project has utilised auto-generative tools in the development of the solutions notebook.

Example prompts used for this project:
- Generate code for a plot
- Create a general README.md template structure for this project
- Generate a doc-string for this function
- Review overall project structure for completeness and consistency
- Present this text data in a clear table below the cell