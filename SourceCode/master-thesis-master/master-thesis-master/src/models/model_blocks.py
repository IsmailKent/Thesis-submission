"""
Basic building blocks for neural nets
"""

import torch.nn as nn

__all__ = ["ConvBlock", "ConvTransposeBlock", "MLPBlock"]


class ConvBlock(nn.Module):
    """ Simple convolutional block for conv. encoders """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, max_pool=None, activation=True):
        """ Module initializer """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if max_pool:
            assert isinstance(max_pool, (int, tuple, list))
            layers.append(nn.MaxPool2d(kernel_size=max_pool, stride=max_pool))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y


class ConvTransposeBlock(nn.Module):
    """ Simple transposed-convolutional block for conv. decoders """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, upsample=None, activation=True, conv_transpose_2d=True):
        """ Module initializer """
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        # adding conv-(bn)-(pool)-act layer
        layers = []
        if conv_transpose_2d:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            )
            # layers.append(nn.Upsample(scale_factor=2))
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if upsample:
            assert isinstance(upsample, (int, tuple, list))
            layers.append(nn.Upsample(scale_factor=upsample))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y


class MLPBlock(nn.Module):
    """
    2-Layer MLP Block with GELU activation function.
    This module is used as defualt MLP on the MLP-Mixer layers.

    Args:
    -----
    input_dim: int
        Input dimensionality
    hidden_dim: int
        Hidden dimensionality in the MLP (out_dim of first layer and in_dim in the second one)
    out_dim: int/None
        Output dimensionality of the MLP. By default it is the same as input_dim
    dropout: float
        Amount of dropout to use after each fully-connected layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.0):
        """ Module Initializer """
        assert isinstance(dropout, float), f"'{dropout = }' must be of type float"
        assert 0 <= dropout and 1 >= dropout, f"{dropout = } must be in range [0, 1]"
        output_dim = output_dim if output_dim is not None else input_dim
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.nonlinearity = nn.GELU()
        if dropout > 0.0:
            self.drop1 = nn.Dropout(dropout)
            self.drop2 = nn.Dropout(dropout)
        else:
            self.drop1, self.drop2 = None, None
        self.act1 = nn.GELU()
        return

    def forward(self, x):
        """ Forward Pass """
        x = self.act1(self.fc1(x))
        x = self.drop1(x) if self.drop1 is not None else x
        x = self.fc2(x)
        y = self.drop2(x) if self.drop2 is not None else x
        return y


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation module. It can be used in the MLP-Mixer predictor

    Args:
    -----
    input_dim: int
        Dimensionality of the input features
    """

    def __init__(self, input_dim):
        """ Module initializer """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.Sigmoid()
        )
        return

    def forward(self, x):
        """ Forward pass """
        B, n_tokens, token_dim = x.shape
        y = self.squeeze(x).view(B, n_tokens)
        y = self.excitation(y).unsqueeze(-1)
        out = x * y.expand_as(x)
        return out


class ResNetBasicBlock(nn.Module):
    """
    Residual convolutional block used to build ResNets following the recipe from SAVI
    In particular, there are two main changes:
      - GroupNorm instead of BatchNorm
      - Stride of 1
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """ Module initalizer """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels)
        )

        self.shortcut = nn.Identity()
        # It used to be:
        # if ResNetBasicBlock.expansion * out_channels:
        if ResNetBasicBlock.expansion * out_channels != in_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                        in_channels,
                        out_channels * ResNetBasicBlock.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                nn.GroupNorm(num_groups=32, num_channels=out_channels * ResNetBasicBlock.expansion)
            )
        self.final_activation = nn.ReLU()
        return

    def forward(self, x):
        """ Forward pass """
        y = self.layers(x) + self.shortcut(x)
        y = self.final_activation(y)
        return y


class Identity(nn.Module):
    """ Custom Identity Module """

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__()

    def forward(self, **kwargs):
        """ """
        y = list(kwargs.values())[0]
        return y


#
