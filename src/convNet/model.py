import logging
from typing import Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.convNet.utils import log_qtiming

logging.basicConfig(level=logging.INFO)


def compute_padding_size(image_dim: int, kernel_size: int, stride: int) -> int:
    """Compute the padding size given an input image of image_dim x image_dim,
        a stride and a filter size"""
    return int(math.ceil(((image_dim - 1) * stride + kernel_size - image_dim) / 2))


def compute_layer_size(image_dim: int, kernel_size: int, stride: int, padding: int=0) -> int:
    return int(math.floor(((image_dim - kernel_size + 2*padding) / stride) + 1))


class convolutional_block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, padding: str="valid", **kwargs) -> None:
        if padding not in ["valid", "same"]:
            raise ValueError("Arg. 'padding' must be one of 'valid' or 'same'")
        if padding == "same":
            if not kwargs.get("image_dim"):
                raise KeyError("If padding set to 'same', you must also pass keyword arg. 'image_dim'")
            self.padding = compute_padding_size(image_dim=kwargs.get("image_dim"), stride=1, kernel_size=kernel_size)
        else:
            self.padding = 0
        super(convolutional_block, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, padding=self.padding)
        self.pool = nn.MaxPool2d(3, 2, padding=1)

    def forward(self, x: Union[torch.tensor, np.array]) -> torch.tensor:
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x


class convNet(nn.Module):
    def __init__(self, dropout=0.1):
        super(convNet, self).__init__()
        self.conv_block_1 = convolutional_block(3, 8, 5, padding="same", image_dim=32)
        self.conv_block_2 = convolutional_block(8, 16, 5)
        self.conv_block_3 = convolutional_block(16, 32, 3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(128, 32)
        self.dense2 = nn.Linear(32, 8)
        self.output = nn.Linear(8, 1)

    @log_timing
    def forward(self, x: Union[torch.tensor, np.array]) -> torch.tensor:
        x = self.conv_block_1(x)
        # Log
        layer_size_conv1 = compute_layer_size(32, self.conv_block_1.kernel_size, 1, self.conv_block_1.padding)
        logging.debug(self.conv_block_1._get_name() + ": size after convolution is %s" % layer_size_conv1)
        layer_size_maxpool1 = compute_layer_size(layer_size_conv1, 3, 2, 1)
        logging.debug(self.conv_block_1._get_name() + ": size after maxpool is %s" % layer_size_maxpool1)

        x = self.conv_block_2(x)
        # Log
        layer_size_conv2 = compute_layer_size(layer_size_maxpool1, self.conv_block_2.kernel_size, 1, self.conv_block_2.padding)
        logging.debug(self.conv_block_2._get_name() + ": size after convolution is %s" % layer_size_conv2)
        layer_size_maxpool2 = compute_layer_size(layer_size_conv2, 3, 2, 1)
        logging.debug(self.conv_block_2._get_name() + ": size after maxpool is %s" % layer_size_maxpool2)

        x = self.conv_block_3(x)
        # Log
        layer_size_conv3 = compute_layer_size(layer_size_maxpool2, self.conv_block_3.kernel_size, 1, self.conv_block_3.padding)
        logging.debug(self.conv_block_3._get_name() + ": size after convolution is %s" % layer_size_conv3)
        layer_size_maxpool3 = compute_layer_size(layer_size_conv3, 3, 2, 1)
        logging.debug(self.conv_block_3._get_name() + ": size after maxpool is %s" % layer_size_maxpool3)

        x = self.flatten(x)
        logging.debug(self._get_name() + ": size after flattening is %s" % (2 * 2 * 32))

        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        y = self.output(x)
        return y
