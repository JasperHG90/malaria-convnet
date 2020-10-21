import logging
from typing import Tuple, Union
import math
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO)


def compute_padding_size(image_dim: int, stride: int, kernel_size: int) -> int:
    """Compute the padding size given an input image of image_dim x image_dim,
        a stride and a filter size"""
    return int(math.ceil(((image_dim - 1) * stride + kernel_size - image_dim) / 2))


def compute_layer_size_conv2d(
    image_dim: int, stride: int, kernel_size: int, padding: int
) -> int:
    return int(((image_dim - kernel_size + 2 * padding) / stride) + 1)


def compute_layer_size_maxpool(image_dim: int, kernel_size: int, stride: int) -> int:
    return int(math.floor(((image_dim - kernel_size) / 2)))


class inception_block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_channels: int,
        output_channels_t1s1: int,
        output_channels_t1s2: int,
        output_channels_t2s1: int,
        output_channels_t2s2: int,
        output_channels_t3s2: int,
        output_channels_t4s1: int,
    ) -> torch.tensor:
        super(inception_block, self).__init__()
        # Set up layers
        self.t1s1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels_t1s1,
            kernel_size=1,
            padding=compute_padding_size(input_dim, 1, 1),
        )
        self.t1s2 = nn.Conv2d(
            in_channels=output_channels_t1s1,
            out_channels=output_channels_t1s2,
            kernel_size=3,
            padding=compute_padding_size(input_dim, 1, 3),
        )
        self.t2s1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels_t2s1,
            kernel_size=1,
            padding=compute_padding_size(input_dim, 1, 1),
        )
        self.t2s2 = nn.Conv2d(
            in_channels=output_channels_t2s1,
            out_channels=output_channels_t2s2,
            kernel_size=5,
            padding=compute_padding_size(input_dim, 1, 5),
        )
        self.t3s1 = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=compute_padding_size(input_dim, 1, 3)
        )
        self.t3s2 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels_t3s2,
            kernel_size=1,
            padding=compute_padding_size(input_dim, 1, 1),
        )
        self.t4s1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels_t4s1,
            kernel_size=1,
            padding=compute_padding_size(input_dim, 1, 1),
        )

    def forward(self, X: Union[torch.tensor, np.array]) -> torch.tensor:
        x1 = torch.relu(self.t1s1(X))
        x1 = torch.relu(self.t1s2(x1))
        x2 = torch.relu(self.t2s1(X))
        x2 = torch.relu(self.t2s2(x2))
        x3 = torch.relu(self.t3s1(X))
        x3 = torch.relu(self.t3s2(x3))
        x4 = torch.relu(self.t4s1(X))
        # Concatenate layers
        x_out = torch.cat([x1, x2, x3, x4], 1)
        return x_out


class convolutional_block(nn.Module):
    def __init__(self, input_dim: int, input_channels: int) -> torch.tensor:
        super(convolutional_block, self).__init__()
        # Set up layers
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=5, stride=2
        )
        # Layer size
        layer_size = compute_layer_size_conv2d(input_dim, 2, 5, 0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        layer_size = compute_layer_size_maxpool(layer_size, 2, 3)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=compute_padding_size(layer_size, 1, 3),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, X: Union[torch.tensor, np.array]) -> torch.tensor:
        x = torch.relu(self.conv1(X))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        return x


class convNet(nn.Module):
    def __init__(self, img_dim: int, dense_units: int, dropout: float = 0.3):
        super(convNet, self).__init__()
        # Set up blocks
        self.conv_block_1 = convolutional_block(input_dim=img_dim, input_channels=3)
        # Compute layer size
        layer_size = compute_layer_size_conv2d(img_dim, 2, 5, 0)
        logging.debug("Layer size after conv_1 is %s" % layer_size)
        layer_size = compute_layer_size_maxpool(layer_size, 2, 3)
        logging.debug("Layer size after maxpool_1 is %s" % layer_size)
        layer_size = compute_layer_size_maxpool(layer_size, 2, 3)
        logging.debug("Layer size after conv_2 and maxpool_2 is %s" % layer_size)
        self.inception_block_1 = inception_block(layer_size, 64, 64, 96, 8, 16, 16, 32)
        self.inception_block_2 = inception_block(
            layer_size, 160, 96, 128, 16, 32, 32, 64
        )
        self.avgpool = nn.AvgPool2d(
            kernel_size=14,
            stride=1,
            # padding = compute_padding_size(layer_size, 2, 3)
        )
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(256, 64)
        self.dense_2 = nn.Linear(64, 16)
        self.dense_3 = nn.Linear(16, 2)
        self.dense_4 = nn.Linear(2, 1)

    # self.dropout = nn.Dropout(dropout)
    # self.batchnorm = nn.BatchNorm1d(dense_units)
    # Output layer
    # self.class_prediction = nn.Linear(dense_units, 1)

    def forward(self, X: Union[torch.tensor, np.array]) -> torch.tensor:
        x = self.conv_block_1(X)
        x = self.inception_block_1(x)
        x = self.inception_block_2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = torch.relu(self.dense_3(x))
        x = self.dense_4(x)
        return x
