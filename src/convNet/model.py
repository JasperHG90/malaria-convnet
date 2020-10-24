import logging
from typing import Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class convolutional_block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, padding: str="valid", **kwargs) -> object:
        if padding not in ["valid", "same"]:
            raise ValueError("Arg. 'padding' must be one of 'valid' or 'same'")
        if padding == "same":
            if not kwargs.get("image_dim"):
                raise KeyError("If padding set to 'same', you must also pass keyword arg. 'image_dim'")
            padding = compute_padding_size(image_dim=kwargs.get("image_dim"), stride=1, kernel_size=kernel_size)
        else:
            padding = None
        super(convolutional_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(3, 2, padding=padding)

    def forward(self, x: Union[torch.tensor, np.array]) -> torch.tensor:
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x


class convNet(nn.Module):
    def __init__(self, dropout=0.1):
        super(convNet, self).__init__()
        self.conv_block_1 = convolutional_block(3, 8, 5)
        self.conv_block_2 = convolutional_block(8, 16, 5)
        self.conv_block_3 = convolutional_block(16, 32, 3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear()
        self.dense2 = nn.Linear()
        self.output = nn.Linear(1)

    def forward(self, x: Union[torch.tensor, np.array]) -> torch.tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        y = self.output(x)
        return y
