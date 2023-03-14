""" adapted from https://github.com/addtt/object-centric-library/blob/d324bb73a91dfa833f313c55a98767734b023503/models/nn_utils.py
    for initializing weights of model """

import logging
import math
import torch
from torch import Tensor, nn



@torch.no_grad()
def init_xavier_(model: nn.Module):
    """Initializes (in-place) a model's weights with xavier uniform, and its biases to zero.
    All parameters with name containing "bias" are initialized to zero.
    All other parameters are initialized with xavier uniform with default parameters,
    unless they have dimensionality <= 1.
    Args:
        model: The model.
    """
    for name, tensor in model.named_parameters():
        if name.endswith(".bias"):
            tensor.zero_()
        elif len(tensor.shape) <= 1:
            pass  # silent
        else:
            torch.nn.init.xavier_uniform_(tensor)
