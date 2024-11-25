#!/bin/bash
#conda activate ModExperiments

PATCHSIZE=3
BATCHSIZE=8192

#Madrid
EPOCHS=880
DATASET='madrid'
DPATH='/home/ragusa/ModelExperiments/datasets/Madrid/hsi/'
GPATH='/home/ragusa/ModelExperiments/datasets/Madrid/gt/'
CHANN=25
SEED=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'WCE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'WCE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'CE-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'CE-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5


########################################################################################
BATCHSIZE=2048

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'WCE-2048-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'WCE-2048-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'CE-2048-ViT-M-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'CE-2048-ViM-M-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5

########################################################################################
# #LAS PALMAS
 PATCHSIZE=3
 BATCHSIZE=8192

 EPOCHS=880
 DATASET='LP'
 DPATH='/home/ragusa/ModelExperiments/datasets/LP/hsi/'
 GPATH='/home/ragusa/ModelExperiments/datasets/LP/gt/'
 CHANN=128
 SEED=0

 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'WCE-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'WCE-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5

 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'CE-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'CE-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5



########################################################################################
 BATCHSIZE=2048

 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8  --nodes 1 --job-name 'WCE-2048-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 8 --nodes 1 --job-name 'WCE-2048-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'weighted_cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5

 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViT --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4  --nodes 2 --job-name 'CE-2048-ViT-LP-P3' --comment 'ViT training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run_with_submitit.py --model ViM --batch-size $BATCHSIZE --epochs $EPOCHS --world-size 8 --num-workers 1 --ngpus 4 --nodes 2 --job-name 'CE-2048-ViM-LP-P3' --comment 'ViM training test' --db-name $DATASET --data-path $DPATH --gt-path $GPATH --patch-size $PATCHSIZE --channels $CHANN --distributed --seed $SEED --criterion 'cross_entropy' --weighted-sampler --sched 'cosine_restart' --decay-rate 1 --warmup-lr 1e-5

########################################################################################


