from typing import Tuple, Union
import math
import torch
import torch.nn as nn
import numpy as np


def compute_padding_size(image_dim: int, stride: int, filter_size: int) -> int:
    """Compute the padding size given an input image of image_dim x image_dim,
        a stride and a filter size"""
    return int(
        math.ceil(((image_dim - 1) * stride + filter_size - (stride * image_dim)) / 2)
    )


class convolutional_block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        maxpool_stride: int = 2,
        maxpool_poolsize: int = 3,
        use_batchnorm: bool = False,
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
            padding=0,
        )
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(
            kernel_size=maxpool_poolsize, stride=maxpool_stride, padding=padding
        )
        # If batch norm
        self._use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm1 = nn.BatchNorm2d(output_channels)
            self.batchnorm2 = nn.BatchNorm2d(output_channels)
            self.batchnorm3 = nn.BatchNorm2d(output_channels)

    def forward(self, X: Union[torch.tensor, np.array]) -> torch.tensor:
        x = torch.relu(self.conv1(X))
        if self._use_batchnorm:
            x = self.batchnorm1(x)
        x = torch.relu(self.conv2(x))
        if self._use_batchnorm:
            x = self.batchnorm2(x)
        s = torch.relu(self.skip(X))
        if self._use_batchnorm:
            s = self.batchnorm3(s)
        # Add layers
        x = x + s
        x = self.maxpool(x)
        return x


class convNet(nn.Module):
    def __init__(
        self, img_dim: Tuple[int, int], dense_units: int, dropout: float = 0.3
    ):
        super(convNet, self).__init__()
        # Set up convolutional blocks
        self.conv_block_1 = convolutional_block(
            3, 32, 3, 1, padding=compute_padding_size(img_dim[0], 1, 3)
        )
        self.conv_block_2 = convolutional_block(
            32, 64, 3, 1, padding=compute_padding_size(int(img_dim[0] / 2), 1, 3)
        )
        self.conv_block_3 = convolutional_block(
            64, 128, 3, 1, padding=compute_padding_size(int(img_dim[0] / 4), 1, 3)
        )
        self.conv_block_4 = convolutional_block(
            128,
            256,
            3,
            2,
            padding=compute_padding_size(int(img_dim[0] / 8), 2, 3),
            use_batchnorm=True,
        )
        self.global_max_pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(dense_units)
        # Output layer
        self.class_prediction = nn.Linear(128, 1)

    def forward(self, X: Union[torch.tensor, np.array]) -> torch.tensor:
        x = self.conv_block_1(X)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = torch.sigmoid(self.class_prediction(x))
        return x
