#%%

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

H = W = 128

from src.convNet.model import inception_block, convolutional_block, convNet
from torchsummary import summary

"""
net = convolutional_block(
    input_dim=H,
    input_channels=3,
)


summary(net, (3, H, W))

H2=W2=14
net = inception_block(
    H2,
    64,
    64,
    96,
    8,
    16,
    16,
    32,
)
summary(net, (64, H2, W2))
"""

net = convNet(H, 32)
# print(net)

from torchsummary import summary

summary(net, (3, H, W))

#%%

from src.convNet.model import compute_padding_size
