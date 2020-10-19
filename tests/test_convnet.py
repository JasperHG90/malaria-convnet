from src.convNet.model import compute_padding_size, convNet


def test_compute_padding_size():
    """Test computing padding size"""

    def output_dim(input_dim, filter_size, padding, stride):
        return int(((input_dim - filter_size + 2 * padding) / stride) + 1)

    D1 = 128
    P = compute_padding_size(D1, 1, 5)
    # Compute output size
    D2 = output_dim(D1, 5, P, 1)
    assert D1 == D2  # noseq
    # Try with larger stride
    P = compute_padding_size(D1, 3, 5)
    D2 = output_dim(D1, 5, P, 2)
    assert (D1 / 2) == D2  # noseq


def test_convNet():
    """Test convNet implementation"""
    CN = convNet(img_dim=(128, 128), dense_units=128, dropout=0.4)
    layers = [layer for layer in CN.named_modules()]
    assert len(layers) == 30  # noseq
