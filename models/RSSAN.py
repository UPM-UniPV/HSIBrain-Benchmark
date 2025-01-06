import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Author: Guillermo Vazquez
email: guillermo.vazquez.valle@upm.es

IMPLEMENTED FOLLOWING:

Residual Spectral–Spatial Attention Network for
Hyperspectral Image Classification

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9103247

"""

class GlobalAveragePool(nn.Module):
	
	def __init__(self, dims):
		
		super().__init__()
		self.dims=dims

	def forward(self, X):
		x_gap = X.mean(self.dims)
		return x_gap


class SpectralAttentionModule(nn.Module):

	def __init__(self, in_feats):

		super().__init__()

		self.mlp = nn.Sequential(nn.Linear(in_feats, in_feats//8),
						   		 nn.ReLU(inplace=True),
								 nn.Linear(in_feats//8, in_feats),
								 nn.ReLU(inplace=True))

	def forward(self, X):

		x_att = self.mlp(X.mean((2,3)))+self.mlp(torch.amax(X, dim=(2,3)))
		return x_att[...,None,None]*X
		

class SpatialAttentionModule(nn.Module):

	def __init__(self, in_feats):

		super().__init__()
		self.conv = nn.Conv2d(2, 1, (1,1))

	def forward(self, X):

		x_att = torch.cat((X.mean(1).unsqueeze(1),torch.amax(X, dim=1).unsqueeze(1)), dim=1)
		return self.conv(x_att)*X
		

class RSSA_Module(nn.Module):

	def __init__(self, in_feats, emb_feats):

		super().__init__()

		self.conv_stage = nn.Sequential(nn.Conv2d(in_feats, emb_feats, 3,
												  padding=1, bias=False),
										nn.BatchNorm2d(emb_feats),
										nn.ReLU(inplace=True),
										nn.Conv2d(emb_feats, emb_feats, 3,
												  padding=1, bias=False),
										nn.BatchNorm2d(emb_feats))
		
		self.SeAM = SpectralAttentionModule(emb_feats)
		self.SaAM = SpatialAttentionModule(emb_feats)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, X):

		x_conv = self.conv_stage(X)
		x_ssa = self.SeAM(self.SaAM(x_conv))
		
		return self.relu(X+x_ssa*x_conv)


class RSSAN(nn.Module):

	def __init__(self, in_dims=1, in_chns=25, embed_dim=32, patch_size=11, out_classes=4):

		super().__init__()

		self.SeAM = SpectralAttentionModule(in_chns)
		self.SaAM = SpatialAttentionModule(in_chns)

		self.conv_block = nn.Sequential(nn.Conv2d(25, embed_dim, 3,
												  padding=1, bias=False),
								  		nn.BatchNorm2d(embed_dim),
										nn.ReLU(inplace=True))
		
		self.SSFL_block = nn.Sequential(RSSA_Module(embed_dim, embed_dim),
										RSSA_Module(embed_dim, embed_dim),
										GlobalAveragePool([2,3]),
										nn.Linear(embed_dim, out_classes))

		
	def forward(self, X):

		x_se = self.SeAM(X)
		x_sa = self.SaAM(x_se)

		return self.SSFL_block(self.conv_block(x_sa))