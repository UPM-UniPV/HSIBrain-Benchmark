#!/bin/bash
#conda activate ModExperiments

export MLFLOW_TRACKING_URI="file:///home/ragusa/ModelExperiments/mlruns_final/"

PATCHSIZE=7 ##3
BATCHSIZE=2048 ##8192

#Madrid
EPOCHS=300 ##250
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25

# FOLD 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'ViT-M-P7' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --easyAtt True --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'ViT+EA-M-P7' --comment 'ViT con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --batch-size $BATCHSIZE --epochs $EPOCHS --caf True --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'SF-M-P7' --comment 'SpectralFormer training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --caf True --easyAtt True --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'SF+EA-M-P7' --comment 'SpectralFormer con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'ViM-M-P7' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HSIMamba --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'HSIMamba-M-P7' --comment 'HSIMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HiT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'HiT-M-P7' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HiT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'HiT-M-P7-lf' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --large-features --seed 0 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type SSMamba --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'SSMamba-M-P7' --comment 'SSMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy'

########################################################################################


# FOLD 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'ViT-M-P7' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --easyAtt True --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'ViT+EA-M-P7' --comment 'ViT con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --batch-size $BATCHSIZE --epochs $EPOCHS --caf True --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'SF-M-P7' --comment 'SpectralFormer training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --caf True --easyAtt True --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'SF+EA-M-P7' --comment 'SpectralFormer con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'ViM-M-P7' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HSIMamba --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'HSIMamba-M-P7' --comment 'HSIMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HiT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'HiT-M-P7' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HiT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'HiT-M-P7-lf' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --large-features --seed 1 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type SSMamba --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'SSMamba-M-P7' --comment 'SSMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy'

########################################################################################


# FOLD 3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'ViT-M-P7' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --easyAtt True --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'ViT+EA-M-P7' --comment 'ViT con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --batch-size $BATCHSIZE --epochs $EPOCHS --caf True --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'SF-M-P7' --comment 'SpectralFormer training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViT --caf True --easyAtt True --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'SF+EA-M-P7' --comment 'SpectralFormer con EasyAttention training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'ViM-M-P7' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HSIMamba --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'HSIMamba-M-P7' --comment 'HSIMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HiT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'HiT-M-P7' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type HiT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'HiT-M-P7-lf' --comment 'HiT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --large-features --seed 2 --criterion 'cross_entropy'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model-type SSMamba --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'SSMamba-M-P7' --comment 'SSMamba training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy'

########################################################################################


