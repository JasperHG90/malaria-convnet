import math
import torch.nn as nn
import torch.nn.functional as F

# Compute padding
def compute_padding_size(image_width, stride, filter_size):
    return math.ceiling(((image_width - 1) * stride - (image_width - filter_size)) / 2)


# Convolutional block
class convolutional_block(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        maxpool_padding,
        maxpool_stride=2,
        maxpool_poolsize=3,
    ):
        super(convolutional_block, self).__init__()
        # Set up layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        # Skip connection
        self.skip = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=stride ** 2,
            padding=padding,
        )
        # Maxpooling layer
        self.maxpool = nn.MaxPool2d(
            kernel_size=maxpool_poolsize, stride=maxpool_stride, padding=maxpool_padding
        )

    def forward(self, X, **kwargs):
        x = F.relu(self.conv1(X))
        x = F.relu(self.conv2(x))
        s = F.relu(self.skip(X))
        # Add layers
        x = x + s
        x = self.maxpool(x)
        return x


# Build convnet model
class convNet(nn.Module):
    def __init__(self):
        self.bla = bla
