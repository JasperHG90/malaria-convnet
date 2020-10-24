#%%

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

H = W = 32

from src.convNet.model import convolutional_block, convNet
from torchsummary import summary

net = convolutional_block(
    input_channels=3,
    output_channels=8,
    kernel_size=5
)

summary(net, (3, H, W))

net = convNet(
    dropout=0.1
)

summary(net, (3, H, W))
