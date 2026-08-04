# Benchmarking Deep Learning Models for Hyperspectral Imaging in Tumor Detection
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)

Evaluation of several deep learning models for brain tumor detection through the use of hyperspectral images from the [SLIM Brain Database](https://slimbrain.citsem.upm.es/) and the [Las Palmas HSI Human Brain Database](https://hsibraindatabase.iuma.ulpgc.es/).


**Table of Contents**
- [Benchmarking Deep Learning Models for Hyperspectral Imaging in Tumor Detection](#benchmarking-deep-learning-models-for-hyperspectral-imaging-in-tumor-detection)
  - [Requirements](#requirements)
  - [Directories and Files](#directories-and-files)
  - [Related Works and Studies](#related-works-and-studies)
  - [Results](#results)
    - [Citation](#citation)
    - [Notes](#notes)

## Requirements
- Pytorch 2.3.1 (+ all the packages in the `.yml` files)
- Cuda 12.4 for parallelization

## Directories and Files
* **job_scripts/** - Contains the .sh files used to launch the training jobs on our clusters.
* **models/** - Contains all the architectures evaluated in this study.
* **plots/** - Conda notebooks used to generate the figures with the results.
* **utils/** - Some useful scripts employed as well as the code for the Focal Loss and the LARC optimizer.
  
The main train/evaluation code is contained in the `main.py` and `engine.py` files. The `run_with_submitit.py` files allow to easily sbatch a job on a Slurm cluster. The json files are the lists of the images from both the dataset used in this work.

## Related Works and Studies
* [The prisma 2020 statement: an updated guideline for reporting systematic reviews](https://doi.org/10.1136/bmj.n71)
* [Hyperspectral Image Classification Using a Hybrid 3D-2D Convolutional Neural Networks](https://doi.org/10.1109/JSTARS.2021.3099118)
* [Ghostnet for hyperspectral image classification](https://doi.org/10.1109/TGRS.2021.3050257)
* [Litedepthwisenet: A lightweight network for hyperspectral image classification](https://doi.org/10.1109/TGRS.2021.3062372)
* [SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers](https://doi.org/10.1109/TGRS.2021.3130716)
* [Classification of hyperspectral image based on double-branch dual-attention mechanism network](https://doi.org/doi:10.3390/rs12030582)
* [Hyperspectral image transformer classification networks](https://doi.org/10.1109/TGRS.2022.3171551)
* [Spectral–spatial attention network for hyperspectral image classification](https://doi.org/10.1109/TGRS.2019.2951160)
* [Residual spectral–spatial attention network for hyperspectral image classification](https://doi.org/10.1109/TGRS.2020.2994057)
* [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929)
* [EfficientNet: Rethinking model scaling for convolutional neural networks](https://arxiv.org/abs/1905.11946)

## Results
All the results obtained from the tests are explained in the following paper: [Benchmarking Deep Learning Models for Hyperspectral Imaging in Tumor Detection](https://doi.org/10.1016/j.cmpb.2026.109571).

### Citation
```bibtex
@article{2026109571,
title = {Benchmarking deep learning architectures for hyperspectral in-vivo brain tumor segmentation},
author = {Guillermo Vazquez and Domenico Ragusa and Emanuele Torti and Elisa Marenzi and Eduardo Juarez and Angel M. Groba and Francesco Leporati},
journal = {Computer Methods and Programs in Biomedicine},
pages = {109571},
year = {2026},
issn = {0169-2607},
doi = {10.1016/j.cmpb.2026.109571}
}
```

### Notes
Please report any issue in the _Issue_ section of the repository. In case of questions or new proposals please report them in the _Discussion_ section.