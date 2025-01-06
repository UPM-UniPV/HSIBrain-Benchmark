import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Author: Guillermo Vazquez
email: guillermo.vazquez.valle@upm.es

IMPLEMENTED FOLLOWING:

LiteDepthwiseNet: A Lightweight Network for
Hyperspectral Image Classification

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9375494

"""

class GroupConvBlock(nn.Module):

	def __init__(self, in_chns, out_chns, k, groups=1):

		super().__init__()

		self.conv = nn.Conv3d(in_chns, out_chns, k, groups=groups, bias=False)
		self.batch_norm = nn.BatchNorm3d(out_chns)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, X):
		return self.relu(self.batch_norm(self.conv(X)))
	

class DWConvBlock(nn.Module):

	def __init__(self, in_chns, out_chns, k, groups=1, pad=0):

		super().__init__()

		self.conv = nn.Conv3d(in_chns, in_chns, k, groups=groups, padding=pad, bias=True)
		self.pw_conv = nn.Conv3d(in_chns, out_chns, 1, groups=1, bias=False)
		self.batch_norm = nn.BatchNorm3d(out_chns)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, X):
		return self.relu(self.batch_norm(self.pw_conv(self.conv(X))))


class GlobalAveragePool(nn.Module):
	
	def __init__(self, dims, transpose_feats=False):
		
		super().__init__()
		self.dims=dims
		self.transpose_feats = transpose_feats

	def forward(self, X):
	
		x_gap = X.mean(self.dims)
		if self.transpose_feats:
			x_gap = torch.transpose(x_gap,1,2).squeeze(1)

		return x_gap


class LiteDwNet(nn.Module):

	def __init__(self, in_dims=1, in_chns=25, patch_size=11, out_classes=4):

		super().__init__()

		self.in_chns = in_chns
		self.patch_size = patch_size

		""""
		FUNCTION TO CALCULATE OUT CHANNELS AT THE END OF BOTH BRANCHES
		-> THERE IS A CONV THAT HAS KERNEL SIZE OF THE TOTAL AMOUNT OF REMAINING CHANNELS
		"""

		self.input_stem = nn.Sequential(nn.Conv3d(in_dims, 24, (7,1,1), bias=False),
								  		nn.BatchNorm3d(24),
										nn.ReLU(inplace=True))
		
		self.left_branch = nn.Sequential(GroupConvBlock(24, 48, 1, groups=3),
										 DWConvBlock(48, 12, (3,3,3), groups=48, pad=1))
		
		self.right_branch = nn.Sequential(GroupConvBlock(24, 48, 1, groups=3),
										  DWConvBlock(48, 12, (3,3,3), groups=48, pad=1),
										  DWConvBlock(12, 12, (3,3,3), groups=12, pad=1))

		shape = self.get_chanel_dim()

		self.joint_branch = nn.Sequential(DWConvBlock(48, 60, (shape[2],3,3), groups=48, pad=(0,1,1)),
										  GlobalAveragePool([3,4], transpose_feats=True),
										  nn.Linear(60, out_classes))


	def get_chanel_dim(self):
		
		x = torch.zeros((1, 1, self.in_chns, self.patch_size, self.patch_size))
		with torch.no_grad():
			x = self.input_stem(x)
		return x.shape
	

	def forward(self, X):

		X = X.unsqueeze(1)

		x = self.input_stem(X)
		x_l = self.left_branch(x)
		x_r = self.right_branch(x)

		x_cat = torch.cat((x_l, x, x_r), dim=1)
		
		x_out = self.joint_branch(x_cat)
		
		return x_out