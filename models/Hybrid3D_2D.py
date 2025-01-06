import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Hyb3D_2D(nn.Module):
	
	def __init__(self, in_dims=1, in_chns=25, patch_size=11, out_classes=4):

		super().__init__()

		self.in_chns = in_chns
		self.patch_size = patch_size

		self.conv_1 = nn.Sequential(nn.Conv3d(in_dims, 32, (3,3,3), bias=False),
					  				nn.BatchNorm3d(32),
									nn.ReLU(inplace=True)
		)

		self.conv_2 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
					  				nn.BatchNorm3d(64),
									nn.ReLU(inplace=True)
		)

		self.conv_3 = nn.Sequential(nn.Conv3d(64, 128, (3,3,3), bias=False),
					  				nn.BatchNorm3d(128),
									nn.ReLU(inplace=True)
		)
		
		self.x1_shape = self.get_shape_after_3dconv()

		self.conv_4 = nn.Sequential(nn.Conv2d(self.x1_shape[1]*self.x1_shape[2], 64, 3, 1, 1, bias=False),
									nn.BatchNorm2d(64),
									nn.ReLU(inplace=True))
		
		self.conv_5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, groups=64, bias=False),
									nn.BatchNorm2d(128),
									nn.ReLU(inplace=True))

		self.x2_shape = self.get_shape_after_2dconv()

		self.dense_1 = nn.Sequential(nn.Linear(self.x2_shape,256),
									nn.ReLU(inplace=True),
									nn.Dropout(p=0.4))
		

		self.dense_2 = nn.Sequential(nn.Linear(256,128),
									 nn.ReLU(inplace=True),
									 nn.Dropout(p=0.4))

		self.dense_out = nn.Linear(128, out_classes)


	def get_shape_after_2dconv(self):
		x = torch.zeros((1, self.x1_shape[1]*self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
		with torch.no_grad():
			x = self.conv_4(x)
			x = self.conv_5(x)
		return x.shape[1]*x.shape[2]*x.shape[3]
	

	def get_shape_after_3dconv(self):
		x = torch.zeros((1, 1, self.in_chns, self.patch_size, self.patch_size))
		with torch.no_grad():
			x = self.conv_1(x)
			x = self.conv_2(x)
			x = self.conv_3(x)
		return x.shape
	
	def forward(self, X):
		
		X = X.unsqueeze(1)
		x = self.conv_1(X)
		x = self.conv_2(x)
		x = self.conv_3(x)
		x = x.view(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4])
		# print(x.shape)
		x = self.conv_4(x)
		x = self.conv_5(x)
		x = x.contiguous().view(x.shape[0], -1)
		# print(x.shape)
		x = self.dense_1(x)
		x = self.dense_2(x)
		out = self.dense_out(x)
		return out