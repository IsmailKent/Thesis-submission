"""
Implementation of decoder modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import build_grid
from models.model_blocks import ConvBlock, ConvTransposeBlock, ResNetBasicBlock

ENCODERS = ["ConvEncoder", "ResNetMini", "ResNetNano", "ResNetMini", "ResNet18", "ResNet34"]
DECODERS = ["ConvDecoder", "UpsampleDecoder", "SpatialBroadCastDecoder", ""]


def get_encoder(encoder_name, downsample_encoder, in_channels, num_channels, kernel_size, **kwargs):
    """ Instanciating an encoder given the model name and parameters """
    if encoder_name not in ENCODERS:
        raise ValueError(f"Unknwon encoder_name {encoder_name}. Use one of {ENCODERS}")

    if(encoder_name == "ConvEncoder"):
        encoder_class = DownsamplingConvEncoder if downsample_encoder else SimpleConvEncoder
        encoder = encoder_class(
                in_channels=in_channels,
                hidden_dims=num_channels,
                kernel_size=kernel_size
            )
    elif(encoder_name == "ResNetNano"):
        encoder = ResNetEncoder(
                block_class=ResNetBasicBlock,
                stages=[1, 1, 0, 0],
                downsample_encoder=downsample_encoder
            )
    elif(encoder_name == "ResNetMini"):
        encoder = ResNetEncoder(
                block_class=ResNetBasicBlock,
                stages=[2, 2, 0, 0],
                downsample_encoder=downsample_encoder
            )
    elif(encoder_name == "ResNet18"):
        encoder = ResNetEncoder(
                block_class=ResNetBasicBlock,
                stages=[2, 2, 2, 2],
                downsample_encoder=downsample_encoder
            )
    elif(encoder_name == "ResNet34"):
        encoder = ResNetEncoder(
                block_class=ResNetBasicBlock,
                stages=[3, 4, 6, 3],
                downsample_encoder=downsample_encoder
            )
    else:
        raise NotImplementedError(f"Unknown encoder {encoder_name}...")

    return encoder


def get_decoder(decoder_name, **kwargs):
    """ Instanciating a decoder given the model name and parameters """
    if decoder_name not in DECODERS:
        raise ValueError(f"Unknwon decoder_name {decoder_name}. Use one of {DECODERS}")

    if(decoder_name == "ConvDecoder"):
        decoder = Decoder(**kwargs)
    elif(decoder_name == "UpsampleDecoder"):
        decoder = UpsampleDecoder(**kwargs)
    elif(decoder_name == "SpatialBroadCastDecoder"):
        decoder = SpatialBroadCastDecoder(
                decoder_resolution=kwargs["decoder_resolution"],
                hidden_dims=kwargs["hidden_dims"]
            )
    elif(decoder_name == ""):
        decoder = DummyDecoder()
    else:
        raise NotImplementedError(f"Unknown decoder {decoder_name}...")

    return decoder


class SimpleConvEncoder(nn.Module):
    """
    Simple fully convolutional encoder

    Args:
    -----
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    """

    def __init__(self, in_channels=3, hidden_dims=(64, 64, 64, 64), kernel_size=5, **kwargs):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.max_pool = kwargs.get("max_pool", None)

        self.encoder = self._build_encoder()
        return

    def _build_encoder(self):
        """ Creating convolutional encoder given dimensionality parameters """
        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                    batch_norm=self.batch_norm,
                    max_pool=self.max_pool
                )
            modules.append(block)
            in_channels = h_dim
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, x):
        """ Forward pass """
        y = self.encoder(x)
        return y


class DownsamplingConvEncoder(nn.Module):
    """
     convolutional encoder that dowsnamples images by factor of 4

    Args:
    -----
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    """

    DOWNSAMPLE = [0, 1, 2]

    def __init__(self, in_channels=3, hidden_dims=(64, 64, 64, 64), kernel_size=5, **kwargs):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.max_pool = kwargs.get("max_pool", None)

        self.encoder = self._build_encoder()
        return

    def _build_encoder(self):
        """ Creating convolutional encoder given dimensionality parameters """
        modules = []
        in_channels = self.in_channels
        # mid = len(self.hidden_dims)//2
        for i, h_dim in enumerate(self.hidden_dims):
            # stride = 2 if i == mid or i == mid+1 else self.stride
            stride = 2 if i in DownsamplingConvEncoder.DOWNSAMPLE else self.stride
            block = ConvBlock(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=self.kernel_size // 2,
                    batch_norm=self.batch_norm,
                    max_pool=self.max_pool
                )
            modules.append(block)
            in_channels = h_dim
        encoder = nn.Sequential(*modules)
        return encoder

    def forward(self, x):
        """ Forward pass """
        y = self.encoder(x)
        return y


class ResNetEncoder(nn.Module):
    """
    Instanciating a ResNet encoder module following the recipe from SAVI
    In particular, there are two main changes:
      - GroupNorm instead of BatchNorm
      - Stride of 1

    Args:
    -----
    block_class: nn Module
        Type of block used for the ResNet layers and stages. Can be either ResNetBasicBlock or Bottleneck
    stages: list
        List with the number of blocks per stage
    """

    NO_STRIDE_STAGES = [0]  # Stages in which strided-convs downsample the feature maps by 2

    def __init__(self, block_class=ResNetBasicBlock, stages=[3, 4, 6, 3], downsample_encoder=False):
        """ ResNet encoder initalizer """
        if len(stages) != 4:
            raise ValueError(f"{stages} must have a length of 4. However, {len(stages) = } != 4")
        super().__init__()
        self.in_channels = 64
        self.block_class = block_class
        self.stages = stages
        self.downsample_encoder = downsample_encoder
        self.out_features = 0

        self.init_conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=32, num_channels=64),
                nn.ReLU()
            )
        self.block_1 = self._make_block(out_channels=64, num_blocks=stages[0], stage_num=0)
        self.block_2 = self._make_block(out_channels=128, num_blocks=stages[1], stage_num=1)
        self.block_3 = self._make_block(out_channels=256, num_blocks=stages[2], stage_num=2)
        self.block_4 = self._make_block(out_channels=512, num_blocks=stages[3], stage_num=3)
        return

    def _make_block(self, out_channels, num_blocks, stage_num):
        """
        Instanciating a ResNet block, which is composed of several basic ResNet blocks
        """
        if num_blocks == 0:
            return nn.Identity()

        layers = []
        for i in range(num_blocks):
            if self.downsample_encoder is False:
                stride = 1
            elif (i < num_blocks - 1 or stage_num in ResNetEncoder.NO_STRIDE_STAGES):
                stride = 1
            else:
                stride = 2
            layers.append(self.block_class(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels * self.block_class.expansion
            self.out_features = out_channels
        block = nn.Sequential(*layers)
        return block

    def forward(self, x):
        """ Forward pass through ResNet encoder """
        output = self.init_conv(x)
        output = self.block_1(output)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        return output


class DummyDecoder(nn.Module):
    """ Dummy decoder that returns output """

    def __init__(self, dims=(35, 35)):
        super().__init__()
        self.dims = dims

    def forward(self, slots):
        B, device = slots.shape[0], slots.device
        return torch.zeros(B, 3, *self.dims).to(device), (None, None, None)


class SpatialBroadCastDecoder(nn.Module):
    """
    Simple convolutional spatial broadcast decoder
    """

    def __init__(self, hidden_dims=(32, 32, 32, 32), decoder_resolution=(35, 35), kernel_size=5, **kwargs):
        """ Decoder intializer """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[-1]
        self.decoder_resolution = decoder_resolution
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.upsample = kwargs.get("upsample", None)

        self.pos_embedding = SoftPositionEmbed(self.out_features, decoder_resolution)
        # self.conv_decoder = Decoder(hidden_dims)
        self.conv_decoder = self._build_conv_decoder()
        return

    def _build_conv_decoder(self):
        """ Building final convolutional part of decoder """

        modules = []
        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvTransposeBlock(
                in_channels=self.hidden_dims[i],
                out_channels=self.hidden_dims[i-1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
                upsample=self.upsample
            )
            modules.append(block)
        # final conv layer
        final_conv = nn.ConvTranspose2d(
                in_channels=self.out_features,
                out_channels=4,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0
            )
        modules.append(final_conv)

        conv_decoder = nn.Sequential(*modules)
        return conv_decoder

    def forward(self, slots):
        """
        Forward pass through full decoder
        """
        B, N, S_DIM = slots.shape

        # adding broadcasing for the dissentangled decoder
        slots = slots.reshape((-1, 1, 1, S_DIM))
        slots = slots.repeat(
                (1, self.decoder_resolution[0], self.decoder_resolution[1], 1)
            )  # slots ~ (B * N_slots, H, W, Slot_dim)

        slots = self.pos_embedding(slots)  # slots ~ (B * N_slots, H, W, Slot_dim)
        slots = slots.permute(0, 3, 1, 2)
        y = self.conv_decoder(slots)  # slots ~ (B * N_slots, Slot_dim, H, W)

        # recons and mask haves shapes [B, N_S, 3 H, W] and [B, N_S, 1, H, W] respectively
        recons, masks = y.reshape(B, -1, 4, y.shape[2], y.shape[3]).split([3, 1], dim=2)

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, (recons, masks)


class Decoder(nn.Module):
    """
    Simple fully convolutional decoder

    Args:
    -----
    in_channels: int
        Number of input channels to the decoder
    hidden_dims: list
        List with the hidden dimensions in the decoder. Final value is the number of output channels
    kernel_size: int
        Kernel size for the convolutional layers
    upsample: int or None
        If not None, feature maps are upsampled by this amount after every hidden convolutional layer
    """

    def __init__(self, in_channels, hidden_dims, kernel_size=5, upsample=None, out_channels=4, **kwargs):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[0]
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.upsample = upsample
        self.out_channels = out_channels

        self.decoder = self._build_decoder()
        return

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.hidden_dims[i]
            modules.append(block)
            if self.upsample is not None and i > 0:
                modules.append(Upsample(scale_factor=self.upsample))
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_channels,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, x):
        y = self.decoder(x)
        return y


class UpsampleDecoder(nn.Module):
    """
    Simple fully convolutional decoder that upsamples by 2 after every convolution

    Args:
    -----
    in_channels: int
        Number of input channels to the decoder
    hidden_dims: list
        List with the hidden dimensions in the decoder. Final value is the number of output channels
    kernel_size: int
        Kernel size for the convolutional layers
    """

    def __init__(self, in_channels, hidden_dims, kernel_size=5, out_channels=4, **kwargs):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = hidden_dims[0]
        self.kernel_size = kernel_size
        self.stride = kwargs.get("stride", 1)
        self.batch_norm = kwargs.get("batch_norm", None)
        self.out_channels = out_channels

        self.decoder = self._build_decoder()
        return

    def _build_decoder(self):
        """
        Creating convolutional decoder given dimensionality parameters
        By default, it maps feature maps to a 5dim output, containing
        RGB objects and binary mask:
           (B,C,H,W)  -- > (B, N_S, 4, H, W)
        """
        modules = []
        in_channels = self.in_channels

        # adding convolutional layers to decoder
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=self.hidden_dims[i],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                batch_norm=self.batch_norm,
            )
            in_channels = self.hidden_dims[i]
            modules.append(block)
            modules.append(Upsample(scale_factor=2))
        # final conv layer
        final_conv = nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_channels,  # RGB + Mask
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        decoder = nn.Sequential(*modules)
        return decoder

    def forward(self, x):
        y = self.decoder(x)
        return y


class Upsample(nn.Module):
    """ Overriding the upsample class to avoid an error of nn.Upsample with large tensors """

    def __init__(self, scale_factor):
        """ Module initializer """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """ Forward pass """
        y = F.interpolate(x.contiguous(), scale_factor=self.scale_factor, mode='nearest')
        return y

    def __repr__(self):
        """ """
        str = f"Upsample(scale_factor={self.scale_factor})"
        return str


class SoftPositionEmbed(nn.Module):
    """ Soft positional embedding with learnable linear projection """

    def __init__(self, hidden_size, resolution, vmin=-1., vmax=1.):
        super().__init__()
        self.projection = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.grid = build_grid(resolution, vmin=-1., vmax=1.).permute(0, 3, 1, 2)
        print(f"resolution: {resolution} , hidden_size: {hidden_size}")

    def forward(self, inputs, channels_last=True):
        """ Projecting grid and adding to inputs """
        b_size = inputs.shape[0]
        if self.grid.device != inputs.device:
            self.grid = self.grid.to(inputs.device)
        grid = self.grid.repeat(b_size, 1, 1, 1)
        emb_proj = self.projection(grid)
        if channels_last:
            emb_proj = emb_proj.permute(0, 2, 3, 1)
        return inputs + emb_proj


class InstanceMaskEncoder(nn.Module):
    """
    Encoder module used to embed the instance segmentation masks for the slot intialization in SAVI.
    The employed module is a simple convolutional encoder with average pooling and an MLP head
    """

    def __init__(self, num_slots, slot_dim, encoder_resolution):
        """ Module initializer """
        super().__init__()
        self.device_param = nn.Parameter(torch.rand(1))  # dummy paramete to get the
        self.conv_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            )
        self.pos_embedding = SoftPositionEmbed(32, encoder_resolution)
        self.encoder_head = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, slot_dim)
            )
        return

    def forward(self, masks):
        """ Forward pass """
        batch_size, seq_len, num_masks, C, H, W = masks.shape
        masks = self._preprocess_masks(masks)
        conv_features = self.conv_encoder(masks)
        augmented_conv_features = self.pos_embedding(conv_features, channels_last=False)
        slots = self.encoder_head(augmented_conv_features)
        slots = slots.view(batch_size, num_masks, -1)  # (B, num_slots, slot_dim)
        return slots

    def _preprocess_masks(self, masks):
        """ Minor preprocessing of the masks prior to encoding them """
        batch_size, seq_len, num_masks, C, H, W = masks.shape
        device = self.encoder_head[-1].weight.device
        masks = masks[:, 0]  # we only want the first time step
        masks = masks.to(device).float()
        masks = masks.contiguous().view(batch_size * num_masks, C, H, W)
        return masks


class InstanceEncoderModule(nn.Module):
    """
    Encoder for Recursive-Instance-based models

    Args:
    -----
    out_backbone_channels: int
        Number of output channels of the backbone model. It will serve as the nuber of input
        channels of the encoder head, and the dimensionality of the positional encoding.
    backbone: nn Module
        Convolutional network serving as backbone to encoder the masks
    encoder_resolution: tuple
        Spatial dimensions of the feature maps to augment with a positional encoding
    slot_dim: int
        Dimensionality of the object slots
    """

    def __init__(self, out_backbone_channels, backbone, encoder_resolution, slot_dim):
        """ Module initializer """
        super().__init__()
        self.backbone = backbone
        self.encoder_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(out_backbone_channels, 256),
                nn.ReLU(),
                nn.Linear(256, slot_dim)
            )
        return

    def forward(self, masks):
        """
        Forward pass

        Args:
        -----
        masks: torch Tensor
            Instance masks in a one-instance-per-channel format. Shape is (B, num_masks, C, H, W)

        Returns:
        --------
        slots: torch Tensor
            Slots corresponding to the encoded instance masks. Shape is (B, num_objs, slot_dim)
        """
        batch_size, num_masks, C, H, W = masks.shape
        masks = masks.reshape(batch_size * num_masks, C, H, W)

        conv_features = self.backbone(masks)
        # augmented_conv_features = self.pos_embedding(conv_features, channels_last=False)
        slots = self.encoder_head(conv_features)
        slots = slots.view(batch_size, num_masks, -1)  # (B, num_slots, slot_dim)
        return slots

#
