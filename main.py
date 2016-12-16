# Written by: Loc Trinh
# Date: 12/16/2016

import matplotlib.pyplot as plt
import numpy as np
import gzip


def get_data():
    """
    Read in image dataset from yann lecun. Normalize to range [0,1].
    Skip offset bytes from header infomation.

    Returns:
        tuple: Tuple of (X_train, Y_train, X_test, Y_test)
    """
    def load_minist_images(filename):
        with gzip.open(filename, 'r') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            num_image = data.shape[0] / 784
        return data.reshape((num_image, 784)) / 255  # normalize pixels

    def load_minist_labels(filename):
        with gzip.open(filename, 'r') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_minist_images("data/train-images-idx3-ubyte.gz")
    Y_train = load_minist_labels("data/train-labels-idx1-ubyte.gz")
    X_test = load_minist_images("data/t10k-images-idx3-ubyte.gz")
    Y_test = load_minist_labels("data/t10k-labels-idx1-ubyte.gz")

    assert X_train.shape == (60000, 784) and Y_train.shape == (60000, )
    assert X_test.shape == (10000, 784) and Y_test.shape == (10000, )

    return (X_train, Y_train, X_test, Y_test)

get_data()
