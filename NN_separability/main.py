import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from mlflow.tracking import MlflowClient
from timm.optim import create_optimizer
from timm.scheduler import CosineLRScheduler, create_scheduler
from torch.utils.data import TensorDataset

from engine import calculate_test_metrics, test_evaluate

import models.extraLayers
import utils.tools as tools
from models.Conv.GhostNet import GhostNet
# from models.Conv.CNN_2D import CNN_2D
from models.DBDA import DBDA
from models.EfficientNet import EfficientNet
from models.HiT import ConvPermuteMLP, HiT
from models.HSIMamba import HSIClassificationMambaModel
from models.Hybrid3D_2D import Hyb3D_2D
from models.LiteDepthwiseNet import LiteDwNet
# from models.mamtrans.MamTrans import MamTrans
from models.RSSAN import RSSAN
from models.SpectralFormer import SpectralFormer
from models.SSAN import SSAN
# from models.ssmamba.ssmamba import mamba_SS_model
# from models.vim.models_mamba import VisionMamba
from models.ViT import ViT
from utils.focal import FocalLoss
from utils.LARC import LARC

matplotlib.use('TkAgg')


# suppressing warning due to lr_scheduler accing current lr
warnings.filterwarnings("ignore", category=UserWarning,
						module='torch.optim.lr_scheduler')
warnings.filterwarnings("ignore", category=DeprecationWarning)
exclude_params = ['inference', 'run_id', 'tracking_uri', 'device', 'data_path', 'gt_path', 'weighted_sampler',
				  'sys_metrics', 'num_workers', 'pin_mem', 'distributed', 'world_size', 'dist_eval', 'ngpus', 'nodes']



def check_dirs(*args):
	for dir_ in args:
		if not os.path.exists(dir_):
			os.makedirs(dir_)

def test_sep(data_loader, model, device, args):

	
	test_preds = []
	test_labels = []
	test_sep_ix = []

	for _, (samples, targets) in enumerate(data_loader, 0):
		samples = samples.to(dtype=torch.float32,
							 device=device, non_blocking=True)
		targets = targets.to(
			dtype=torch.long, device=device, non_blocking=True)

		torch.cuda.synchronize()

		model.sep_forward(samples, targets)

		model.eval()
		with torch.no_grad():
			outputs, sep_ixs = model.sep_forward(samples, targets)

		test_preds.append(outputs.cpu())
		test_labels.append(targets.cpu())
		test_sep_ix.append(sep_ixs)

	test_preds = torch.cat(test_preds)
	test_labels = torch.cat(test_labels)

	test_sep_ix = np.vstack(test_sep_ix)

	print(test_sep_ix.mean(axis=0))

	torch.cuda.synchronize()

	if args.distributed and (args.dist_eval is True):
		dist.barrier()
		test_preds = tools.gather_tensor(test_preds)
		test_labels = tools.gather_tensor(test_labels)

	# # remove background
	# mask = (test_labels != 0)
	# test_preds_noback = test_preds[mask]
	# test_preds_noback = test_preds_noback - 1
	# test_labels = test_labels[mask] - 1


	kappa_score, precision, recall, f1, accuracy, roc_auc, cm, per_class_accuracy, precision_class, recall_class, fscore_class, roc_per_class, support = calculate_test_metrics(
		test_preds, test_labels)

	print(f1)
	print(accuracy)
	print(cm)
	# quit()

	return test_sep_ix.mean(axis=0)

def get_args_parser():

	parser = argparse.ArgumentParser(
		'Training and evaluation script', add_help=False)

	parser.add_argument('--inference', action='store_true',
						default=False, help='Perform inference only - no training')
	parser.add_argument('--run-id', default='4b5f59d9505a466bb6fd381b2fe993e1',
						type=str, help='MLFlow Run ID to be used for inference')
	parser.add_argument('--tracking-uri',
						default="file://" + os.getcwd() + "/mlruns_final_complete",
						type=str, help='MLFlow tracking URI'
						)

	# Basic parameters
	parser.add_argument('--model-type', default='MamTrans',
						type=str, help='Model type (default: "ViT")')
	parser.add_argument('--batch-size', default=512, type=int,
						help='Batch size')  # 8192 to be used with LARS
	parser.add_argument('--epochs', default=1, type=int,
						help='Total epochs to run')
	parser.add_argument('--device', default='cuda',
						help='Device to use for training / testing')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--job_name', default="", type=str)

	# ViT parameters
	parser.add_argument('--mlp-dim', default=4, type=int,
						help='Number of features in the mlp')
	parser.add_argument('--heads', default=16, type=int,
						help='Number of heads in the attention layers')
	parser.add_argument('--easyAtt', default=False, type=bool,
						help='Use EasyAttention instead of Attention')
	parser.add_argument('--caf', default=False, type=bool, help='Use CAF')
	parser.add_argument('--dropPath-rate', default=0.1,
						type=float, help='DropPath rate')

	# ViT/ViM parameters
	parser.add_argument('--blocks', default=4, type=int,
						help='Number of blocks in the transformer')
	parser.add_argument('--patch-size', default=7, type=int, help='Patch size')
	parser.add_argument('--embed-dim', default=64,
						type=int, help='Embeddings dimension')
	parser.add_argument('--classes', default=4, type=int,
						help='Number of classes to predict (default: 4)')
	parser.add_argument('--drop', type=float, default=0.2,
						metavar='PCT', help='Dropout rate (default: 0.1)')

	# HSIMamba parameters
	parser.add_argument('--deltat', default=0.01, type=float,
						help='Delta parameter for HSIMamba')
	parser.add_argument('--output-dim', default=128, type=int,
						help='Output dimension of the HSIMamba model')

	# HiTMamba parameters
	parser.add_argument('--large-features', action='store_true',
						default=False, help='Use a higher number of features')

	# Optimizer parameters
	parser.add_argument('--opt', default='adamw', type=str,
						metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
	parser.add_argument('--use-larc', default=True, type=bool, help='Use LARC')
	parser.add_argument('--criterion', default='focal',
						type=str, help='Criterion (default: "focal")')
	parser.add_argument('--opt-eps', default=1e-8, type=float,
						metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
	parser.add_argument('--opt-betas', default=None, type=float, nargs='+',
						metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
	parser.add_argument('--weight-decay', type=float,
						default=5e-5, help='weight decay (default: 5e-5)')
	parser.add_argument('--momentum', type=float, default=0.9,
						metavar='M', help='SGD momentum (default: 0.9)')

	# Focal loss parameters
	parser.add_argument('--gamma', type=float, default=2,
						help='Gamma parameter for focal loss (default: 2)')

	# Learning rate schedule parameters
	parser.add_argument('--sched', default='cosine', type=str,
						metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
	parser.add_argument('--lr', type=float, default=1e-3,
						metavar='LR', help='Learning rate (default: 1e-3)')
	parser.add_argument('--lr-noise', type=float, nargs='+', default=None,
						metavar='pct, pct', help='Learning rate noise on/off epoch percentages')
	parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
						help='Learning rate noise limit percent (default: 0.67)')
	parser.add_argument('--lr-noise-std', type=float, default=1.0,
						metavar='STDDEV', help='Learning rate noise std-dev (default: 1.0)')
	parser.add_argument('--warmup-lr', type=float, default=1e-6,
						metavar='LR', help='Warmup learning rate (default: 1e-6)')
	parser.add_argument('--t-initial', type=int, default=50,
						help='Initial T value for cosine scheduler')
	parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
						help='Lower lr bound for cyclic schedulers that hit 0 (1e-5)')
	parser.add_argument('--decay-epochs', type=float, default=10,
						metavar='N', help='Epoch interval to decay LR')
	parser.add_argument('--cycle-mul', type=float, default=1.3,
						metavar='N', help='Cycle multiplier for cosine restarts')
	parser.add_argument('--cycle-limit', type=int, default=7,
						metavar='N', help='Cycle limit for cosine restarts')
	parser.add_argument('--cycle-decay', type=int, default=0.9,
						metavar='N', help='Cycle decay for cosine restarts')
	parser.add_argument('--warmup-epochs', type=int, default=5,
						metavar='N', help='Epochs to warmup LR, if scheduler supports')
	parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
						help='Epochs to cooldown LR at min_lr, after cyclic schedule ends')
	parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
						help='Patience epochs for Plateau LR scheduler (default: 10')
	parser.add_argument('--decay-rate', '--dr', type=float, default=0.1,
						metavar='RATE', help='LR decay rate (default: 0.1)')

	# Dataset parameters
	parser.add_argument('--db-name',
						# default='LP',
						default='M',
						type=str, help='Dataset name')

	parser.add_argument('--data-path', default='/path/to/hsi_npy/',
						type=str, help='Dataset path')
	parser.add_argument('--gt-path', default='/path/to/gt_map/',
						type=str, help='GT path')

	parser.add_argument('--gdv_save_path', default='/path/to/gdv_vect/',
							type=str, help='Separability index save path')


	parser.add_argument('--readable_path', default='/path/to/mlruns_readable/',
							type=str, help='Path generated through folder_translate.py')
		
	
	parser.add_argument('--channels', type=int, default=128,
						help='Number of channels in the dataset')
	parser.add_argument('--train-pcg', default='0.7',
						type=float, help='Train set split percentage')
	parser.add_argument('--val-pcg', default='0.1', type=float,
						help='Validation set split percentage')
	parser.add_argument(
		'--densify-labels', default=[2, 3], nargs='+', type=int, help="Labels to densify")
	parser.add_argument(
		'--augment-labels', default=[2, 3], nargs='+', type=int, help="Labels to augment")
	parser.add_argument('--weighted-sampler', action='store_true',
						default=False, help='Use a weighted sampler')

	# Distributed training parameters
	parser.add_argument('--distributed', action='store_true',
						default=False, help='Enabling distributed training')
	parser.add_argument('--world-size', default=1, type=int,
						help='Number of distributed processes')
	parser.add_argument('--dist-eval', action='store_true',
						default=False, help='Enabling distributed evaluation')

	# Mlflow parameters
	parser.add_argument('--sys-metrics', default=False,
						type=bool, help='Log system metrics')

	# Other parameters
	parser.add_argument('--num-workers', default=1, type=int)
	parser.add_argument('--pin-mem', action='store_true',
						help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
	parser.add_argument('--no-pin-mem', action='store_false',
						dest='pin_mem', help='')
	parser.set_defaults(pin_mem=True)

	return parser


def select_model(args):
	# By default the hyperparameters are the ones used for the original papers
	if (args.model_type == 'ViT'):
		model = ViT(patchSize=args.patch_size, nBlocks=args.blocks, mlp_dim=args.mlp_dim, caf=args.caf, easyAtt=args.easyAtt, numHeads=args.heads,
					embedDim=args.embed_dim, numClasses=args.classes, dropout=args.drop, dropPath=args.dropPath_rate, channels=args.channels)
	elif args.model_type == 'EfficientNet':
		model = EfficientNet(in_dims=args.channels, out_classes=args.classes)
	elif args.model_type == 'DBDA':
		model = DBDA(band=args.channels, classes=args.classes)
	elif args.model_type == 'SpectralFormer':
		model = SpectralFormer(patch_size=args.patch_size,
							   num_patches=args.channels, num_classes=args.classes)
	elif args.model_type == 'SSAN':
		model = SSAN(patch_size=args.patch_size,
					 num_band=args.channels, n_classes=args.classes)
	elif args.model_type == 'GhostNet':
		model = GhostNet(input_channels=args.channels, n_classes=args.classes)
	elif args.model_type == 'Hyb3D_2D':
		model = Hyb3D_2D(in_chns=args.channels,
						 patch_size=args.patch_size, out_classes=args.classes)
	elif args.model_type == 'RSSAN':
		model = RSSAN(in_chns=args.channels,
					  patch_size=args.patch_size, out_classes=args.classes)
	elif args.model_type == 'LiteDwNet':
		model = LiteDwNet(in_chns=args.channels,
						  patch_size=args.patch_size, out_classes=args.classes)
	elif args.model_type == 'HiT':
		if args.db_name == 'M':
			if args.large_features:
				embed_dims = [128, 128, 256, 256]
			else:
				embed_dims = [56, 56, 88, 88]
		elif args.db_name == 'LP':
			if args.large_features:
				embed_dims = [536, 536, 640, 640]
			else:
				embed_dims = [256, 256, 512, 512]
		model = nn.Sequential(
			models.extraLayers.AddDimensionLayer(1),
			HiT(layers=[4, 3, 14, 3], img_size=args.patch_size, in_chans=args.channels, num_classes=args.classes,
				embed_dims=embed_dims, transitions=[False, True, False, False], segment_dim=[8, 8, 4, 4], mlp_ratios=[3, 3, 3, 3], skip_lam=1.0,
				qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
				# Doesn't work for single pixels, only for patches to capturre
				# spatial information
				norm_layer=nn.LayerNorm, mlp_fn=ConvPermuteMLP, large_features=args.large_features)
		)
	else:
		print('Model not found')
		exit()

	return model


def main(args):
	# os.environ['MLFLOW_TRACKING_URI'] = args.tracking_uri
	# # os.environ["MLFLOW_TRACKING_URI"] = "file:///home/ragusa/HSIBrain/TEST"
	# # #TEST

	# tools.init_distributed_mode(args)
	# temp_dir = Path("./tmp")

	device = torch.device(args.device)

	# if args.distributed:
	#     args.batch_size = int(args.batch_size / (args.ngpus * args.nodes))
	#     args.num_workers = int(
	#         (args.num_workers + (args.ngpus * args.nodes) - 1) / (args.ngpus * args.nodes))


	base_gdv_path = args.gdv_save_path


	# fix the seed
	seed = args.seed
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


	cudnn.benchmark = True
	torch.cuda.empty_cache()

	with open(f'image_list_{args.db_name}.json', 'r') as f:
		image_list = json.load(f)

	""" RANDOM SPLITTING """
	train_val_ids = []
	tumor_IDs, nontumor_IDs = tools.get_tumor_IDs(image_list, args.gt_path)

	random.Random(seed).shuffle(tumor_IDs)
	random.Random(seed).shuffle(nontumor_IDs)

	T_train_ids, T_val_ids, T_test_ids = tools.random_split(tumor_IDs, args.train_pcg,
															args.val_pcg, seed)

	train_ids, validation_ids, test_ids = tools.random_split(nontumor_IDs, args.train_pcg,
															 args.val_pcg, seed)

	train_ids.extend(T_train_ids)
	validation_ids.extend(T_val_ids)
	test_ids.extend(T_test_ids)

	train_val_ids.extend(train_ids)
	train_val_ids.extend(validation_ids)
	""" ********* """

	min_vect, max_vect = tools.min_max_norm_val(
		args.data_path, args.gt_path, train_val_ids, args.channels)
	

	base_path = args.readable_path
	model_path = f"{base_path}{args.db_name}/{args.model_type}/{seed}/model/model.pth"
	# model_path = f"{base_path}{args.db_name}/{args.model_type}/{seed}/model/"

	model = select_model(args)

	model = torch.load(model_path, weights_only=False,
					   map_location="cpu")
	
	if args.model_type == 'HiT':
		model = model[1]


	# model = mlflow.pytorch.load_model(model_path)
	# model.load_state_dict(torch.load(model_path, weights_only=False))

	model.to(device)

	test_data, test_labels, test_lab_count_noDens, _ = tools.loadImagesData(
			args.data_path, args.gt_path, test_ids, patch_size=args.patch_size, labelsToDensify=[], labelsToAugment=[], minMaxVects=[min_vect, max_vect])


	test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
	test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)

	# test_data = test_data[test_labels>0,...]
	# test_labels = test_labels[test_labels>0]

	dataset_test = TensorDataset(test_data, test_labels)

	if args.dist_eval:
		sampler_test = torch.utils.data.DistributedSampler(
			dataset_test, num_replicas=tools.get_world_size(), rank=tools.get_rank(), shuffle=False)
	else:
		sampler_test = torch.utils.data.SequentialSampler(dataset_test)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, sampler=sampler_test,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True,
		shuffle=None
	)

	gdv_vect = test_sep(data_loader_test, model, device, args)


	""" SAVE GDV  VECTORS """

	save_path = f"{base_gdv_path}{args.db_name}/{args.model_type}/"
	check_dirs(save_path)
	print(save_path)
	
	np.save(f"{save_path}gdv_vect_{args.seed}.npy", gdv_vect)
	
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		'Training and Evaluation Script', parents=[get_args_parser()])
	args = parser.parse_args()

	main(args)