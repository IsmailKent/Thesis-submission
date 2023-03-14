"""
SAVi, but instance based, rather than image based. Therefore, no need for Slot Attention
"""

import torch
import torch.nn as nn

from models import SoftPositionEmbed, InstanceEncoderModule, get_encoder, get_decoder, get_initalizer
from models.model_blocks import Identity


class RecursiveInstanceEncoder(nn.Module):
    """
    Recurseve encoder module to embed object instances from a video into informative
    embedding representation (e.g. Slots)

    Args:
    -----
    resolution: list/tuple (int, int)
        spatial size of the input images
    num_slots: integer
        number of object slots to use. Corresponds to N-objects + background
    encoder_type: string
        Name of the encoder to use
    hidden_dims: list/tuple of integers (int, ..., int)
        number of output channels for each conv layer
    decoder_resolution: list/tuple (int, int)
        spatial resolution of the decoder. If not the same as 'resolution', the
        decoder needs to use some padding/stride for upsampling
    initializer: string
        Type of intializer employed to initialize the slots at the first time step
    """

    def __init__(self, resolution, num_slots, slot_dim=64, in_channels=1, kernel_size=5,
                 encoder_type="ConvEncoder", num_channels=(32, 32, 32, 32), downsample_encoder=False,
                 downsample_decoder=False, upsample=4, decoder_resolution=(8, 8), use_predictor=True,
                 initializer="LearnedRandom", **kwargs):
        """ Module initializer """
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        self.decoder_resolution = decoder_resolution
        self.initializer_mode = initializer
        self.use_predictor = use_predictor

        print("Initializer:")
        print(f"  --> mode={initializer}")
        print(f"  --> slot_dim={slot_dim}")
        print(f"  --> num_slots={num_slots}")
        self.initializer = get_initalizer(
                mode=initializer,
                slot_dim=slot_dim,
                num_slots=num_slots
            ) if self.use_predictor else Identity()

        # Building convolutional encoder modules
        print("Encoder:")
        print(f"  --> Encoder_type={encoder_type}")
        print(f"  --> Downsample_encoder={downsample_encoder}")
        print(f"  --> in_channels={in_channels}")
        print(f"  --> num_channels={num_channels}")
        print(f"  --> kernel_size={kernel_size}")
        backbone = get_encoder(
                encoder_name=encoder_type,
                downsample_encoder=downsample_encoder,
                in_channels=in_channels,
                num_channels=num_channels,
                kernel_size=kernel_size
            )
        out_features = num_channels[-1] if encoder_type in ["ConvEncoder"] else backbone.out_features
        self.encoder = InstanceEncoderModule(
                out_backbone_channels=out_features,
                backbone=backbone,
                encoder_resolution=resolution,
                slot_dim=slot_dim
            )

        # predictor that fuses observation with state
        print("Fusion Model:")
        print(" --> Slot Dim")
        self.fusion_module = FusionModule(
                slot_dim=slot_dim
            ) if self.use_predictor else Identity()

        # Building decoder modules
        print("Decoder:")
        print(f"  --> Resolution={resolution}")
        print(f"  --> Num channels={num_channels}")
        print(f"  --> Upsample={upsample}")
        print(f"  --> Downsample_encoder={downsample_encoder}")
        print(f"  --> Downsample_decoder={downsample_decoder}")
        print(f"  --> Decoder_resolution={decoder_resolution}")
        self.decoder_pos_embedding = SoftPositionEmbed(
                hidden_size=slot_dim,
                resolution=self.decoder_resolution
            )
        num_channels = kwargs.get("num_channels_decoder", num_channels[:-1])
        self.decoder = get_decoder(
                decoder_name="UpsampleDecoder",
                in_channels=slot_dim,
                hidden_dims=num_channels,
                kernel_size=kernel_size,
                upsample=True,
                out_channels=in_channels
            )
        return

    def forward(self, input, num_imgs=10, num_slots=None, **kwargs):
        """ Forward pass through the model """
        slot_history = []
        reconstruction_history = []

        predicted_slots = self.initializer(batch_size=input.shape[0], **kwargs)
        for t in range(num_imgs):
            imgs = input[:, t]
            slots = self.encoder(imgs)
            # recons = self.decode(slots)
            predicted_slots = self.fusion_module(encoded_slots=slots, state=predicted_slots)
            recons = self.decode(predicted_slots)
            slot_history.append(slots)
            reconstruction_history.append(recons)

        slot_history = torch.stack(slot_history, dim=1)
        recons_history = torch.stack(reconstruction_history, dim=1)
        return slot_history, recons_history

    def decode(self, slots):
        """ Decoding slots into object representations """
        B, N_S, S_DIM = slots.shape

        # adding broadcasing for the dissentangled decoder
        slots = slots.reshape((-1, S_DIM, 1, 1))
        slots = slots.repeat((1, 1, self.decoder_resolution[0], self.decoder_resolution[1]))

        # decoding and reshaping
        y = self.decoder(slots)  # slots ~ (B * N_slots, Slot_dim, H, W)
        recons = y.reshape(B, -1, self.in_channels, y.shape[2], y.shape[3])
        return recons


class FusionModule(nn.Module):
    """
    Fuses the information between the slot representations extracted from the current
    object instance masks and the slot representations from the previous time step.
    """

    def __init__(self, slot_dim):
        """ """
        super().__init__()
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        return

    def forward(self, encoded_slots, state):
        """
        Fusing slots-state with slots extracted from the current observation

        Args:
        -----
        encoded_slots: torch Tensor
            Encoded object instances. Shape is (B, num_slots, slot_dim)
        slot_dim: torch Tensor
            Slots from the previous iteration. Shape is (B, num_slots, slot_dim)

        Returns:
        --------
        slots: torch Tensor
            Fused slots. Shape is (B, num_slots, slot_dim)
        """
        B, N, D = encoded_slots.shape
        slots = self.gru(
                encoded_slots.reshape(B * N, D),
                state.reshape(B * N, D)
            )
        slots = slots.reshape(B, N, D)
        slots = slots + encoded_slots
        return slots
