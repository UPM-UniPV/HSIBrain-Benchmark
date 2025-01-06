# Benchmarking Deep Learning Models for Hyperspectral Imaging in Tumor Detection
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)

This is an evaluation of several deep learning models for brain tumor detection through the use of hyperspectral images from the [SLIM Brain Database](https://slimbrain.citsem.upm.es/) and the [Las Palmas HSI Human Brain Database](https://hsibraindatabase.iuma.ulpgc.es/).


**Table of Contents**
- [Benchmarking Deep Learning Models for Hyperspectral Imaging in Tumor Detection](#benchmarking-deep-learning-models-for-hyperspectral-imaging-in-tumor-detection)
  - [Requirements](#requirements)
  - [Directories and files](#directories-and-files)
  - [Related Works and Studies](#related-works-and-studies)
  - [Results](#results)
    - [Citation](#citation)

## Requirements
- Pytorch 2.3.1 (+ all the packages in the requirements.txt file)
- Cuda 12.4 for parallelization

## Directories and files
* **models/** - Contains all the code for the models evaluated in this study.
* **utils/** - There are some utils functions as well as the code for the Focal Loss and the LARC optimizer and other scripts used for evaluating the results.
* **job_scripts/** - Contains the .sh files used to launch the training jobs on our clusters.
  
The main train/evaluation code is contained in the `main.py` and `engine.py` files. The `run_with_submitit.py` files allow to easily sbatch a job on a Slurm cluster. The json files are the lists of the images from both the dataset used in this work.

## Related Works and Studies
* []()
* []()
* []()

## Results
:dart: All the results obtained from the tests are explained in the following paper: [Benchmarking Deep Learning Models for Hyperspectral Imaging in Tumor Detection]().

### Citation
```bibtext

```