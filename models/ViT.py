import torch
import torch.nn as nn

from timm.models.layers import DropPath

'''
MLP block composed of two linear layers with GELU activation and dropout.
'''
class FeedForward(nn.Module):
    def __init__(self, embedDim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embedDim, mlp_dim*embedDim)
        self.act = nn.Tanh()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim*embedDim, embedDim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
'''
Attention block implemented as in https://arxiv.org/abs/2010.11929
'''
class Attention(nn.Module):
    def __init__(self, embedDim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.headDim = embedDim // heads
        self.scale = self.headDim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(embedDim, embedDim * 3)
        self.proj = nn.Linear(embedDim, embedDim)

        #same value for both the dropout layers
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, GS, _ = x.shape
        
        qkv = self.qkv(x)

        qkv = (
            self.qkv(x)
            .reshape(B, GS, 3, self.heads, self.headDim)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

"""
Easy Attention as https://arxiv.org/pdf/2308.12874
"""
class DenseEasyAttn(nn.Module):
    
    def __init__(self,  d_model, 
                        seqLen,
                        num_head,) -> None:
        """
        Dense Easy attention mechansim used in transformer model for the time-series prediction and reconstruction
        
        Args:

            d_model     :   The embedding dimension for the input tensor 
            
            seqLen      :   The length of the sequence 

            num_head    :   The number of head to be used for multi-head attention
    
        """
        super(DenseEasyAttn,self).__init__()
     
        assert d_model % num_head == 0, "dmodel must be divible by number of heads"

        self.d_model    =   d_model
        self.d_k        =   d_model // num_head
        self.num_heads  =   num_head
        # Create the tensors
        self.Alpha      = nn.Parameter(torch.randn(size=(num_head,seqLen+1,d_model),dtype=torch.float),requires_grad=True)  #seqLen+1          
        self.WV         = nn.Parameter(torch.randn(size=(d_model,seqLen+1)       ,dtype=torch.float),requires_grad=True) #dmodel            
        # Initialisation
        nn.init.xavier_uniform_(self.Alpha)
        nn.init.xavier_uniform_(self.WV)
    
    def split_heads(self, x):
        """
        Split the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, S, N]
        
        Returns:
            x   : sequence with shape = [B, H, S, N//H]
        """
        batch_size, seq_length, d_model = x.size()
        return x.contiguous().view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine the sequence into multi-heads 

        Args:
            x   : Input sequence shape = [B, H, S, N//H]
        
        Returns:
            x   : sequence with shape = [B, S, N]
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
 
    def forward(self,x:torch.Tensor):   
        """
        Forward prop for the easyattention module 
        
        Following the expression:  x_hat    =   Alpha @ Wv @ x 

        Args:  
            
            self    :   The self objects

            x       :   A tensor of Input data
        
        Returns:
            
            x       :   The tensor be encoded by the moudle
        
        """
        # Obtain the value of batch size 
        B,_,_   =   x.shape
        # We repeat the tensor into same number of batch size that input has 
        Wv      =   self.WV.repeat(B,1,1)    
        # Implement matmul along the batch 
        V       =   torch.bmm(Wv,x)
        # Split heads for value
        V_h     =   self.split_heads(V)
        # Implement the learnable attention tensor 
        Alpha   =   self.Alpha.repeat(B,1,1,1)
        # Gain Attention by matrix multiplication
        x       =   Alpha @ V_h
        # Combine the output back to the original size
        x       =   self.combine_heads(x)
        
        return x

'''
Transformer block composed of an attention block and an MLP block plus normalization layers.
'''
class TBlock(nn.Module):
    def __init__(self, embedDim, heads, mlp_dim, channels, easyAtt, dropout, dropPath):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedDim)
        self.norm2 = nn.LayerNorm(embedDim)

        self.dropPath = DropPath(dropPath) if dropPath > 0. else nn.Identity()

        if easyAtt == False:
            self.attn = Attention(embedDim, heads, dropout)
        else:
            self.attn = DenseEasyAttn(embedDim, channels, heads)

        self.mlp = FeedForward(embedDim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.dropPath(self.attn(self.norm1(x)))
        x = x + self.dropPath(self.mlp(self.norm2(x)))

        return x

'''
Weights initialization function.
If the layer is a linear layer, the weights are initialized using a truncated normal distribution with a standard deviation of 0.02. Eventually the bias is set to 0.
If the layer is a layer normalization layer, the weights are initialized to 1 and the bias to 0.
'''
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        #nn.init.constant_(m.weight, 1.0) #DEBUG
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

'''
Vision Transformer for hyperspectral brain images classification/segmentation, by default the images have a size of 128x128
and a number of bands equal to 25. The classes are [tumor, vein, artery, normal, duraMater].
'''
class ViT (nn.Module):
    def __init__(self, patchSize, nBlocks, mlp_dim, numHeads, caf=False, easyAtt=False, embedDim=64, numClasses=4, dropout=0.1, dropPath=0.1, channels=25):
        super().__init__()

        self.patchSize = patchSize
        self.nBlocks = nBlocks
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.numClasses = numClasses
        self.caf = caf

        self.patchEmbed = nn.Linear(patchSize*patchSize, embedDim)

        #Learned class token and position embeddings
        self.clsToken = nn.Parameter(torch.zeros(1, 1, embedDim))
        self.posEmbed = nn.Parameter(torch.zeros(1, channels+1, embedDim))

        # Initialization of the additional embeddings using a truncated normal distribution
        nn.init.trunc_normal_(self.clsToken.data, std=0.02)
        nn.init.trunc_normal_(self.posEmbed.data, std=0.02)

        self.dropout = nn.Dropout(dropout)

        #Set of transformer encoders
        self.blocks = nn.ModuleList([TBlock(embedDim, numHeads, mlp_dim, channels, easyAtt, dropout, dropPath) for _ in range(nBlocks)])
        
        #CAF
        if self.caf != False:
            self.skipcat = nn.ModuleList([])
            for _ in range(nBlocks-2):
                self.skipcat.append(nn.Conv2d(channels+1, channels+1, [1, 2], 1, 0))

        self.norm = nn.LayerNorm(embedDim)
        self.mlpHead = nn.Linear(embedDim, numClasses)

        self.apply(init_weights)

    def forward(self, px):
        B, C, H, W = px.shape

        #Patch embedding
        x = self.patchEmbed(px.reshape(B, C, H*W))

        #Add clsToken
        clsToken = self.clsToken.expand(B, -1, -1)
        x = torch.cat((clsToken, x), dim=1)

        #Add positional embeddings
        posEmbed = self.posEmbed
        x = x + posEmbed
        x = self.dropout(x)

        #Transformer blocks (+CAF), adapted from https://arxiv.org/pdf/2107.02988.pdf
        if self.caf == False:
            for blk in self.blocks:
                x = blk(x)
        else:
           last_output = []
           nb = 0
           for blk in self.blocks:           
               last_output.append(x)
               if nb > 1:  
                   x = self.skipcat[nb-2](torch.cat([x.unsqueeze(3), last_output[nb-2].unsqueeze(3)], dim=3)).squeeze(3)
               x = blk(x)
               nb += 1

        x = x[:, 0] # get clsToken
        
        x = self.norm(x)
        x = self.mlpHead(x) #return the class logits for the pixel

        return x