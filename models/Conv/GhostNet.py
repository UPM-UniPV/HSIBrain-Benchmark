import torch
import torch.nn as nn
import math

class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out = channel // reduction
        if out == 0:
            out = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, out),
            nn.ReLU(inplace=True),
            nn.Linear(out, channel),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

    
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]    
    

class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

    
    
    
class GhostNet(nn.Module):

    def __init__(self, input_channels, n_classes, dropout=True):
        super(GhostNet, self).__init__()
        
        self.stem = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.2)
            
        self.BottleNeck1 = GhostBottleneck(16, 16, 16, 1, 1, True)
        self.BottleNeck2 = GhostBottleneck(16, 48, 24, 3, 1, True)
        self.BottleNeck3 = GhostBottleneck(24, 72, 24, 1, 1, True)
        
        self.conv5 = nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(72)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(72, 216)
        self.bn3 = nn.BatchNorm1d(216)
        self.fc2 = nn.Linear(216, n_classes)


    def forward(self, x):

        x1 = self.stem(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if self.use_dropout:
            x1 = self.dropout(x1)

        x2 = self.BottleNeck1(x1)
        x2 = self.BottleNeck2(x2)
        x2 = self.BottleNeck3(x2)
        
        x3 = self.conv5(x2)
        x3 = self.bn2(x3)
        x3 = self.relu(x3)
        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc1(x3)
        x3 = self.bn3(x3)
        if self.use_dropout:
            x1 = self.dropout(x1)        
        x3 = self.fc2(x3)
        
        return x3