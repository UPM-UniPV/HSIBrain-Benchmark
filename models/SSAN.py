import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalDot(nn.Module):
    def __init__(self, in_channels, depth, embed=True, softmax=True, maxpool=False):
        """
        Implementation of the non-local block in PyTorch.

        Args:
            in_channels (int): Number of input channels of the input tensor.
            depth (int): The number of channels in which to execute the non-local operation.
            embed (bool): Whether to use the "embedded" version as in the paper (Sec. 3.2).
            softmax (bool): Whether to apply softmax for soft attention.
            maxpool (bool or int): Max-pooling kernel size. Use `False` for no max-pooling.
        """
        super(NonLocalDot, self).__init__()
        self.embed = embed
        self.softmax = softmax
        self.maxpool = maxpool

        if self.embed:
            self.embA = nn.Conv2d(in_channels, depth, kernel_size=1, stride=1)
            self.embB = nn.Conv2d(in_channels, depth, kernel_size=1, stride=1)
        else:
            self.embA = self.embB = None

        self.g = nn.Conv2d(in_channels, depth, kernel_size=1, stride=1)

        self.out_conv = nn.Conv2d(depth, in_channels, kernel_size=1, stride=1)
        self.out_bn = nn.BatchNorm2d(in_channels)
        nn.init.constant_(self.out_bn.weight, 0)  # Zero-init as described in Sec. 4.1


    def forward(self, X):
        """
        Args:
            X (Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            Tensor: Output tensor of the same shape as the input.
        """
        batch_size, _, height, width = X.size()

        # Embedding
        if self.embed:
            a = self.embA(X)
            b = self.embB(X)
        else:
            a, b = X, X

        g = self.g(X)

        # Optional max pooling
        if self.maxpool:
            pool = nn.MaxPool2d(kernel_size=self.maxpool, stride=self.maxpool)
            b = pool(b)
            g = pool(g)

        # Flatten
        a_flat = a.view(batch_size, -1, height * width)  # [B, depth, HW]
        b_flat = b.view(batch_size, -1, b.size(2) * b.size(3))  # [B, depth, HW]
        g_flat = g.view(batch_size, -1, g.size(2) * g.size(3))  # [B, depth, HW]

        # Compute f(a, b) -> [B, HW, HW]
        f = torch.bmm(a_flat.permute(0, 2, 1), b_flat)
        if self.softmax:
            f = F.softmax(f, dim=-1)
        else:
            f = f / f.size(-1)

        # Compute f * g -> [B, HW, depth]
        fg = torch.bmm(f, g_flat.permute(0, 2, 1))
        fg = fg.permute(0, 2, 1).contiguous().view(batch_size, -1, height, width)

        # Residual connection and final batch norm
        fg = self.out_conv(fg)
        fg = self.out_bn(fg)
        return X + fg
    

# Define the network
class SSAN(nn.Module):
    def __init__(self, patch_size=11, num_band=25, n_classes=4):
        super(SSAN, self).__init__()
        self.patch_size = patch_size
        self.num_band = num_band
        self.n_classes = n_classes

        # Spectral feature learning
        self.conv3d_1 = nn.Conv3d(1, 32, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0))
        self.conv3d_2 = nn.Conv3d(32, 32, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0))

        # Spatial feature learning
        self.nonlocal_1 = NonLocalDot(32 * num_band, 64)
        self.conv2d_1 = nn.Conv2d(32 * num_band, 64, kernel_size=3, padding=1)

        self.nonlocal_2 = NonLocalDot(64, 64)
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.nonlocal_3 = NonLocalDot(64, 64)

        # Classification layers
        self.fc1 = nn.Linear(64 * patch_size * patch_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        # Spectral feature learning
        # x = x.permute(0, 3, 1, 2)  # Transpose to [batch, num_band, patch_size, patch_size]
        x = x.unsqueeze(1)         # Add channel dimension [batch, 1, num_band, patch_size, patch_size]
        x1 = F.relu(self.conv3d_1(x))
        x2 = F.relu(self.conv3d_2(x1))

        # Spatial feature learning
        x3 = x2.permute(0, 3, 4, 1, 2)  # Transpose to [batch, patch_size, patch_size, num_band, 32]
        x3 = x3.reshape(-1, self.patch_size, self.patch_size, 32 * self.num_band)  # Flatten the spectral dimension
        x3 = self.nonlocal_1(x3.permute(0, 3, 1, 2))  # Non-local layer, permute to [batch, channels, height, width]
        x4 = F.relu(self.conv2d_1(x3))

        x4 = self.nonlocal_2(x4)
        x5 = F.relu(self.conv2d_2(x4))

        x5 = self.nonlocal_3(x5)

        # Classification
        x5 = x5.view(x5.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x5))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
