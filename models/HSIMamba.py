"""
Model from https://arxiv.org/pdf/2404.00272
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HSIVimBlock(nn.Module):
    def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init):
        super(HSIVimBlock, self).__init__()
        # Initialization with self.hidden_dim
        self.spatial_dim = spatial_dim
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim

        # LayerNorm is now expecting a flattened feature vector of Bands*H*W elements
        self.norm = nn.LayerNorm(num_bands * spatial_dim * spatial_dim)

        # Adjust linear layers according to the new input dimension
        self.linear_x = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)
        self.linear_z = nn.Linear(num_bands * spatial_dim * spatial_dim, hidden_dim)

        self.forward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.backward_conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        #self.C = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.delta_param = nn.Parameter(torch.full((hidden_dim,), delta_param_init))

        self.linear_forward = nn.Linear(hidden_dim, output_dim)
        self.linear_backward = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        Batch, H, W, Bands = x.shape  # Correct shape extraction assuming [Batch, Height, Width, Bands]

        # Correctly reshape for LayerNorm to flatten all spatial and spectral information
        x = x.reshape(Batch, -1)  # New shape: [Batch, Bands*H*W]

        # Normalize across the flattened spatial-spectral data
        x = self.norm(x)

        # Projection to hidden dimensions
        x_proj = self.linear_x(x)
        z_proj = self.linear_z(x)

        # Ensure correct reshaping for Conv1d compatibility
        x_proj = x_proj.contiguous().view(Batch, self.hidden_dim, -1)
        z_proj = z_proj.contiguous().view(Batch, self.hidden_dim, -1)

        # Reverse z_proj for the backward path
        z_proj_reversed = torch.flip(z_proj, dims=[-1])

        # Bidirectional Conv1d processing using reversed input for the backward path
        x_forward = F.silu(self.forward_conv1d(x_proj))
        x_backward = F.silu(self.backward_conv1d(z_proj_reversed))

        # Apply delta parameter correctly
        delta_expanded = self.delta_param.unsqueeze(0).unsqueeze(2)  # Correct shape for broadcasting

        # SSM processing with delta applied, using the original and reversed inputs for forward and backward paths respectively
        forward_ssm_output = torch.tanh(self.forward_conv1d(x_proj) + self.A * delta_expanded)
        backward_ssm_output = torch.tanh(self.backward_conv1d(z_proj_reversed) + self.B * delta_expanded)

        # Combine forward and backward outputs into a single representation
        forward_reduced = forward_ssm_output.mean(dim=2)
        backward_reduced = backward_ssm_output.mean(dim=2)

        # Combine the reduced forward and backward paths
        y_forward = self.linear_forward(forward_reduced)
        y_backward = self.linear_backward(backward_reduced)

        # Element-wise sum of forward and backward outputs
        y_combined = y_forward + y_backward

        # Return the combined output
        return y_combined
    
class SpatialFeatureProcessing(nn.Module):
    def __init__(self, input_channels):
        super(SpatialFeatureProcessing, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional layer with dilation rate of 1 (standard convolution)
            nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=(3, 3), padding=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Second convolutional layer with a higher dilation rate to increase the receptive field
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=2, dilation=2),  # Note the increased padding to maintain the spatial dimensions
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adding global average pooling

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        return x
    
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        # Remove softmax here if you're using a loss function that includes it, such as nn.CrossEntropyLoss
        return x
    
class HSIClassificationMambaModel(nn.Module):
    def __init__(self, spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init, num_classes):
        super(HSIClassificationMambaModel, self).__init__()
        self.vim_block = HSIVimBlock(spatial_dim, num_bands, hidden_dim, output_dim, delta_param_init)
        self.output_dim = output_dim  # Save output_dim as an attribute of the class

        # Initialize SpatialFeatureProcessing and Classifier here
        # Adjusted to pass 'output_dim' as 'input_channels' to SpatialFeatureProcessing
        self.spatial_processing = SpatialFeatureProcessing(input_channels=output_dim)
        # Assuming the output of SpatialFeatureProcessing matches the in_features expected by Classifier
        self.classifier = Classifier(in_features=512, num_classes=num_classes)

    def forward(self, x):
        x = self.vim_block(x)
        # This is a placeholder. Actual reshaping depends on the output of HSIVimBlock and the input expectation of SpatialFeatureProcessing
        x = x.contiguous().view(-1, self.output_dim, 1, 1)  # Reshape to include spatial dimensions if needed
        x = self.spatial_processing(x)

        # Flatten the output from spatial processing if it's not already flat
        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)
        return x