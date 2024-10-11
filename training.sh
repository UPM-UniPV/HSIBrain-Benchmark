#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size 8192 --epochs 100 --world-size 8 --num_workers 32 --ngpus 4 --nodes 2 --job-name 'ViT' --comment 'ViT training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --easyAtt True --batch-size 8192 --epochs 100 --world-size 8 --num_workers 32 --ngpus 8 --nodes 1 --job-name 'ViT+EA' --comment 'ViT con EasyAttention training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size 8192 --epochs 100 --caf True --world-size 8 --num_workers 32 --ngpus 4 --nodes 2 --job-name 'SpectralF' --comment 'SpectralFormer training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --caf True --easyAtt True --batch-size 8192 --epochs 100 --world-size 8 --num_workers 32 --ngpus 8 --nodes 1 --job-name 'SpectralF+EA' --comment 'SpectralFormer con EasyAttention training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size 8192 --epochs 100 --world-size 8 --num_workers 32 --ngpus 4 --nodes 2 --job-name 'ViM' --comment 'ViM training test'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model HSIMamba --batch-size 8192 --epochs 100 --world-size 8 --num_workers 32 --ngpus 8 --nodes 1 --job-name 'HSIMamba' --comment 'HSIMamba training test'
