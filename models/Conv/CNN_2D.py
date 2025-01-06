import torch
import torch.nn as nn


class Stem(nn.Module):

	def __init__(self, in_dim, out_dim):
		
		super().__init__()
		self.conv = nn.Conv2d(in_dim, out_dim, 3)		
		self.ln = nn.LayerNorm(out_dim)
		
	def forward(self, X):
		x = self.conv(X).permute(0, 2, 3, 1)
		return self.ln(x).permute(0, 3, 1, 2)


class DSBlock(nn.Module):

	def __init__(self, in_dim, out_dim):
		
		super().__init__()
		self.ln = nn.LayerNorm(in_dim)
		self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)		

	def forward(self, X):
		x = X.permute(0, 2, 3, 1)
		x = self.ln(x).permute(0, 3, 1, 2)
		return self.conv(x)


class ConvNeXtBlock(nn.Module):

	def __init__(self, dim):
		
		super().__init__()

		self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) # depthwise conv
		self.norm = nn.LayerNorm(dim)
		self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
		self.act = nn.GELU()
		self.pwconv2 = nn.Linear(4 * dim, dim)

	def forward(self, X):

		x_in = X
		x = self.dwconv(X)
		x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
		x = self.norm(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		
		x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
		x = x_in + x
		return x
	
    
class OutBlock(nn.Module):

	def __init__(self, in_dim, out_dim):
		
		super().__init__()	
		self.ln = nn.LayerNorm(in_dim)
		self.linear = nn.Linear(in_dim, out_dim)	
		
	def forward(self, X):
		return self.linear(self.ln(X.mean([-2, -1])))
	

class CNxtN_2D(nn.Module):

	"""
	INTENDED TO WORK WITH 11X11 PATCHES
	"""

	def __init__(self, in_dims=25, out_classes=4):

		super().__init__()

		# self.conv_1 = ConvNeXtBlock()
		self.stem = Stem(in_dims,96)

		self.stage_1 = nn.Sequential(ConvNeXtBlock(96),
					  				 ConvNeXtBlock(96),
					  				 ConvNeXtBlock(96))
		self.ds_1 = DSBlock(96, 192)

		self.stage_2 = nn.Sequential(ConvNeXtBlock(192),
					  				 ConvNeXtBlock(192),
					  				 ConvNeXtBlock(192),
									 ConvNeXtBlock(192),
									 ConvNeXtBlock(192),
									 ConvNeXtBlock(192))
		self.ds_2 = DSBlock(192, 384)
		
		self.stage_3 = nn.Sequential(ConvNeXtBlock(384),
					  				 ConvNeXtBlock(384),
					  				 ConvNeXtBlock(384))
		
		self.out = OutBlock(384, out_classes)
		

		"""
		ends up with AvgPool + LN + Linear(n_classes)
		"""

		self.initialize_weights()

	def initialize_weights(self):

		for m in self.modules():

			if isinstance(m, nn.Conv2d):
				# Initializing Conv2d layers with Kaiming Normal initialization
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				# Initializing BatchNorm2d layers with a small positive constant
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				# Initializing Linear layers with Kaiming Normal initialization
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)


	def forward(self, X):

		x = self.stem(X)
		x = self.stage_1(x)
		x = self.ds_1(x)
		
		x = self.stage_2(x)
		x = self.ds_2(x)
		
		x = self.stage_3(x)
				
		return self.out(x)
