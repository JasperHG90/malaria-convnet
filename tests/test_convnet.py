import math
from src.convNet.model import compute_padding_size, convNet, convolutional_block
import torch


def compute_layer_size(image_dim: int, kernel_size: int, stride: int, padding: int=0) -> int:
    return int(math.floor(((image_dim - kernel_size + 2*padding) / stride) + 1))


def test_compute_padding_size():
    """Test computing padding size"""

    D1 = 32
    P = compute_padding_size(D1, 5, 1)
    # Compute output size
    D2 = compute_layer_size(D1, 5, 1, P)
    assert D1 == D2  # noseq
    # Try with larger stride
    P = compute_padding_size(D1, 5, 2)
    D2 = compute_layer_size(D1, 5, 2, P)
    assert D1 == D2  # noseq


def test_convolutional_block():
    """Test inception block"""
    CB = convolutional_block(3, 6, 5, "valid")
    # Compute layer sizes
    size_after_conv2d = compute_layer_size(32, 5, 1, 0)
    assert size_after_conv2d == 28
    size_after_maxpool = compute_layer_size(28, 3, 2, 1)
    assert size_after_maxpool == 14
    # Make tensor
    x = torch.rand(2, 3, 32, 32)
    x = CB(x)
    # Shape should be 2, 6, 14, 14
    assert tuple(x.shape) == (2, 6, 14, 14)
    # Test padding = "same"
    CB = convolutional_block(3, 6, 5, "same")
    x = torch.rand(2, 3, 32, 32)
    x = CB(x)
    assert tuple(x.shape) == (2, 6, 16, 16)



def test_convNet():
    """Test convNet implementation"""
    CN = convNet(dropout=0.3)
