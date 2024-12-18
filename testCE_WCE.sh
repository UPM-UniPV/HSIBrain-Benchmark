#!/bin/bash
#conda activate ModExperiments

PATCHSIZE=3
BATCHSIZE=8192

#Madrid
EPOCHS=300
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25

##FOLD 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'noRes-WCE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'weighted_cross_entropy' --weighted-sampler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'noRes-WCE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'weighted_cross_entropy' --weighted-sampler

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'noRes-CE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy' --weighted-sampler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'noRes-CE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 0 --criterion 'cross_entropy' --weighted-sampler

##FOLD 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'noRes-WCE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'weighted_cross_entropy' --weighted-sampler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'noRes-WCE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'weighted_cross_entropy' --weighted-sampler

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'noRes-CE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy' --weighted-sampler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'noRes-CE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 1 --criterion 'cross_entropy' --weighted-sampler

##FOLD 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'noRes-WCE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'weighted_cross_entropy' --weighted-sampler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'noRes-WCE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'weighted_cross_entropy' --weighted-sampler

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'noRes-CE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy' --weighted-sampler
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'noRes-CE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed 2 --criterion 'cross_entropy' --weighted-sampler


########################################################################################
#BATCHSIZE=2048

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'noRes-WCE-2048-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'noRes-WCE-2048-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'noRes-CE-2048-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'noRes-CE-2048-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler

########################################################################################
# #LAS PALMAS
 #PATCHSIZE=3
 #BATCHSIZE=8192

 #EPOCHS=300
 #DATASET='LP'
 #DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
 #GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
 #CHANN=128
 #SEED=0

 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'noRes-WCE-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler
 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'noRes-WCE-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler

 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'noRes-CE-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler
 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'noRes-CE-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler



########################################################################################
 #BATCHSIZE=2048

 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'noRes-WCE-2048-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler
 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'noRes-WCE-2048-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler
 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'noRes-CE-2048-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler 
 #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'noRes-CE-2048-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler

########################################################################################


