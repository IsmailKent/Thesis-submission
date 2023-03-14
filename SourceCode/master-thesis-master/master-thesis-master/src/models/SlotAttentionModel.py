"""
Implementation of the Slot Attention model for image decomposition
    - Locatello, Francesco, et al. "Object-centric learning with slot attention." NeurIPS 2020
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SlotAttention, SoftPositionEmbed, get_decoder, SimpleConvEncoder, DownsamplingConvEncoder
from models.nn_utils import init_xavier_


class SlotAttentionModel(nn.Module):
    """
    SlotAttention model as described in the paper:
        - Locatello, Francesco, et al. "Object-centric learning with slot attention." NeurIPS 2020

    Args:
    -----
    resolution: list/tuple (int, int)
        spatial size of the input images
    num_slots: integer
        number of object slots to use. Corresponds to N-objects + background
    num_iterations: integer
        numbe rof recurrenSlotAttentionModelt iterations for slot refinement
    in_channels: integer
        number of input (e.g., RGB) channels
    kernel_size: integer
        side of the CNN filters
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    decoder_resolution: list/tuple (int, int)
        spatial resolution of the decoder. If not the same as 'resolution', the
        decoder needs to use some padding/stride for upsampling
    """

    def __init__(self, resolution, num_slots, slot_dim=64, num_iterations=3, in_channels=3,
                 kernel_size=5, num_channels=(32, 32, 32, 32), downsample_encoder=True, upsample=4,
                 decoder_resolution=(8, 8), **kwargs):
        """ Model initializer """
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.hidden_dims = num_channels
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]
        mlp_hidden = kwargs.get("mlp_hidden", 128)
        print(downsample_encoder)

        # Building encoder modules
        if downsample_encoder:
            self.encoder = DownsamplingConvEncoder(
                    in_channels=in_channels,
                    hidden_dims=num_channels,
                    kernel_size=kernel_size
                )
        else:
            self.encoder = SimpleConvEncoder(
                    in_channels=in_channels,
                    hidden_dims=num_channels,
                    kernel_size=kernel_size
                )
        print("Decoder:")
        print(f"  --> Resolution={resolution}")
        print(f"  --> Num channelsl={num_channels}")
        print(f"  --> Upsample={upsample}")
        print(f"  --> Downsample_encoder={downsample_encoder}")
        print(f"  --> Decoder_resolution={decoder_resolution}")

        self.encoder_pos_embedding = SoftPositionEmbed(
                hidden_size=self.out_features,
                resolution=decoder_resolution if downsample_encoder else resolution,
                vmin=0.,
                vmax=1.
            )
        self.encoder_mlp = nn.Sequential(
            nn.LayerNorm(self.out_features),
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Building decoder modules
        self.decoder_pos_embedding = SoftPositionEmbed(
                hidden_size=slot_dim,
                resolution=self.decoder_resolution,
                vmin=0.,
                vmax=1.
            )

        if downsample_encoder:
            self.decoder = get_decoder(
                    decoder_name="ConvDecoder",
                    in_channels=slot_dim,
                    hidden_dims=num_channels,
                    kernel_size=kernel_size,
                    decoder_resolution=decoder_resolution,
                    upsample=upsample
                )
        else:
            self.decoder = get_decoder(
                decoder_name="ConvDecoder",
                in_channels=slot_dim,
                hidden_dims=num_channels[:-1],
                kernel_size=kernel_size
            )

        self.slot_attention = SlotAttention(
            dim_feats=self.out_features,
            dim_slots=slot_dim,
            num_slots=self.num_slots,
            num_iters=self.num_iterations,
            mlp_hidden=mlp_hidden,
        )
        self._init_model()
        return

    # adapted from: https://github.com/addtt/object-centric-library/blob/main/models/slot_attention/trainer.py
    @torch.no_grad()
    def _init_model(self):
        init_xavier_(self)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_ih)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_hh)
        torch.nn.init.orthogonal_(self.slot_attention.gru.weight_hh)
        limit = math.sqrt(6.0 / (1 + self.slot_attention.dim_slots))
        torch.nn.init.uniform_(self.slot_attention.slots_mu, -limit, limit)
        torch.nn.init.uniform_(self.slot_attention.slots_sigma, -limit, limit)

    def forward(self, input, num_slots=None):
        """ Forward pass through the model """
        slots = self.encode(input, num_slots=num_slots)
        slot_embs = slots

        recon_combined, (recons, masks) = self.decode(slots)

        return recon_combined, (recons, masks, slot_embs)

    def encode(self, input, num_slots=None):
        """ Encoding an image into slots """
        B, C, H, W = input.shape

        # encoding input and adding positional encodding
        x = self.encoder(input)  # x ~ (B,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos_embedding(x)  # x ~ (B,H,W,C)

        # further encodding with 1x1 Conv (implemented as shared MLP)
        x = torch.flatten(x, 1, 2)
        x = self.encoder_mlp(x)  # x ~ (B, N, Dim)

        # slot module
        slots = self.slot_attention(x, num_slots=num_slots)  # slots ~ (B, N_slots, Slot_dim)

        return slots

    def decode(self, slots):
        """ Decoding slots into object representations """
        B, N_S, S_DIM = slots.shape

        # adding broadcasing for the dissentangled decoder
        slots = slots.reshape((-1, 1, 1, S_DIM))
        slots = slots.repeat(
                (1, self.decoder_resolution[0], self.decoder_resolution[1], 1)
            )  # slots ~ (B*N_slots, H, W, Slot_dim)

        # adding positional embeddings to reshaped features
        slots = self.decoder_pos_embedding(slots)  # slots ~ (B*N_slots, H, W, Slot_dim)
        slots = slots.permute(0, 3, 1, 2)

        y = self.decoder(slots)  # slots ~ (B*N_slots, Slot_dim, H, W)

        # recons and masks have shapes [B, N_S, C, H, W] & [B, N_S, 1, H, W] respectively
        y_reshaped = y.reshape(B, -1, self.in_channels + 1, y.shape[2], y.shape[3])
        recons, masks = y_reshaped.split([self.in_channels, 1], dim=2)

        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)

        return recon_combined, (recons, masks)


#
