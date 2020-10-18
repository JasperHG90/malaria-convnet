import torch.nn as nn

# Convolutional block
class convolutional_block(nn.Module):
    def __init__(self, filters, kernel_size, stride, maxpool_padding = "same", maxpool_stride = 2, maxpool_poolsize = 3):
        super(convolutional_block, self).__init__()
        # Set up layers
        self.conv1 = nn.Conv2d


# Build convnet model
# TODO: padding should happen here
class convNet(nn.Module):
    def __init__(self):
