# AML & DAAI 2024/2025 Project - Task Arithmetic Under Bias-Variance Trade-offs


## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.8, installed and correctly configured on your system. 

(The base code of this project has been produced using CUDA 11.8 and Python 3.10.9)

Follow https://pytorch.org/get-started/locally/ to setup PyTorch (note that PyTorch comes pre-installed on Google Colab's environments, so you can skip this step)

Once you have properly setup everything, make sure you are in the correct directory and run from the command line:
```bash
pip install -r requirements.txt
```


## Base Code Structure
The starting code should already provide everything needed to easily extend it. Read carefully the specifications in the project report.

In the following, you can find a brief description of the included files.

| File/Folder | Description |
| ---- | ----------- |
| `args.py` | contains the function responsible for parsing each command line argument. |
| `datasets/` | contains the files with code to load data, build splits and dataloaders. |
| `utils.py` | contains several utilities to correctly setup the pipeline. |
| `task_vectors.py` | contains the code for building task vectors and/or load checkpoints. |
| `modeling.py` | contains the backbone architectures and modules used in the project. |
| `heads.py` | contains the logic to build and store the open-vocabulary classifiers used in the project. |
| `finetune.py` |  It contains the function responsible for finetuning the pretrained model | 
| `finetune_balance.py` | It contains the functions responsible for class balancing and finetuning on balanced class   | 
| `eval_single_task.py` | It contains the code for evaluating a single checkpoints or all checkpoints for all dataset in a folder | 
| `eval_task_addition.py` | It contsins the functions for adding task vectors and creating multi task architecture  | 
| `eval.py` |  It contains some utility functions for evaluating multi task model | 
| `balanced_data.py` | It contains some utility functions for constructing the balanced datasets  | 
## Running The Experiments
In order to run the experiments we modified the variables inside the code itself and didn't use the scripts to run the code.