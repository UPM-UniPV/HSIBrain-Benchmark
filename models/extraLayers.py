import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patchSize, embedDim=64):
        super().__init__()
        self.patchEmbed = nn.Linear(patchSize*patchSize, embedDim)

    def forward (self, x):
        B, C, H, W = x.shape
        return self.patchEmbed(x.reshape(B, C, H*W))
    
class PermuteLayer(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class AddDimensionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.unsqueeze(self.dim)
        return x