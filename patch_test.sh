#!/bin/bash

source /home/guillermo.vazquez/env/bin/activate

python3 main.py --model-type DBDA --device cuda:6 --epochs 200 --patch-size 7 --batch-size 2048
python3 main.py --model-type SpectralFormer --device cuda:6 --epochs 200 --patch-size 7 --batch-size 2048
python3 main.py --model-type SSAN --device cuda:6 --epochs 200 --patch-size 7 --batch-size 2048
python3 main.py --model-type GhostNet --device cuda:6 --epochs 200 --patch-size 7 --batch-size 2048
python3 main.py --model-type Hyb3D_2D --device cuda:6 --epochs 200 --patch-size 7 --batch-size 2048