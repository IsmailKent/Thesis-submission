"""
Implementation of predictor modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Callable
from torch import Tensor
from lib.logger import print_

from models.model_blocks import MLPBlock, SEBlock


class LSTMPredictor(nn.Module):
    """
    LSTM for predicting the (n+1)th object slot given a sequence of n slots

    Args:
    -----
    slot_dim: integer
        dimensionality of the slots
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        determine number of lstm cells
    mode: string
        intialization of the states
    residual: bool
        If True, a residual connection bridges across the predictor module
    """

    def __init__(self, slot_dim=64, hidden_dim=64, num_layers=2, mode="zeros", residual=True):
        """ Module initializer """
        assert mode in ["zeros", "random", "learned"]
        super().__init__()
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mode = mode
        self.residual = residual

        self.lstm = nn.ModuleList([])
        for n in range(num_layers):
            dim = slot_dim if n == 0 else hidden_dim
            self.lstm.append(nn.LSTMCell(input_size=dim, hidden_size=hidden_dim))

        self.init_hidden()
        return

    def forward(self, x):
        """
        Forward pass through model

        Args:
        -----
        x: torch Tensor
            Current sequence element fed to the RNN. Shape is (B, Dim)

        Returns:
        --------
        output: torch Tensor
            Predicted next element in the sequence. Shape is (B, Dim)
        """
        input = x
        for i in range(self.num_layers):
            h, c = self.hidden_state[i]
            next_h, next_c = self.lstm[i](input, (h, c))
            self.hidden_state[i] = (next_h, next_c)
            input = self.hidden_state[i][0]

        output = input + x if self.residual else input
        return output

    def init_hidden(self, b_size=1, device=None):
        """ Initializing hidden and cell states """
        hidden_state = []
        for _ in range(self.num_layers):
            cur_state = (torch.zeros(b_size, self.hidden_dim), torch.zeros(b_size, self.hidden_dim))
            hidden_state.append(cur_state)
        if device is not None:
            hidden_state = [(h[0].to(device), h[1].to(device)) for h in hidden_state]
        self.hidden_state = hidden_state
        return


class VanillaTransformerPredictor(nn.Module):
    """
    Transformer Predictor module

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256,
                 num_layers=2, n_heads=4, residual=False):
        """ Module initializer """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        print_("Instanciating Vanilla Transformer Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")

        # MLPs to map slot-dim into token-dim and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim is split across num_heads, i.e., each head will have dimension embed_dim // num_heads)
        self.transformer_encoders = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )
        return

    def get_attn_maps(self, inputs):
        """ Computing the attention maps for the given inputs """
        B, num_imgs, num_slots, slot_dim = inputs.shape
        mask, _ = self.get_mask_pattern(num_imgs, num_slots)
        mask = mask.to(inputs.device)

        # slots to tokens
        token_input = self.mlp_in(inputs)

        # Applying encoding to inform transformer about timestamp of each object
        # it will be applied time dimension (enforced with max_len = self.num_imgs)
        time_pos_encoding = PositionalEncoding(
                batch_size=B,
                num_slots=num_slots,
                d_model=self.token_dim,
                max_len=num_imgs
            ).to(inputs.device)
        time_encoded_input = time_pos_encoding(token_input).reshape(B, num_imgs * num_slots, self.token_dim)

        # feeding through transformer encoder blocks minus last one
        token_output = time_encoded_input
        for encoder in self.transformer_encoders[:-1]:
            token_output = encoder(token_output, src_mask=mask)

        # feeding through last transformer layer, fetching the attention maps.
        last_encoder_layer = self.transformer_encoders[-1]
        norm_output = last_encoder_layer.norm1(token_output)
        attn_out, attn_weights = last_encoder_layer.self_attn(
                norm_output,
                norm_output,
                norm_output,
                attn_mask=mask,
                key_padding_mask=None,
                need_weights=True
            )
        return attn_weights

    def forward(self, inputs):
        """ Foward pass """
        B, num_imgs, num_slots, slot_dim = inputs.shape
        mask, _ = self.get_mask_pattern(num_imgs, num_slots)
        mask = mask.to(inputs.device)

        # slots to tokens
        token_input = self.mlp_in(inputs)

        # Applying encoding to inform transformer about timestamp of each object
        # it will be applied time dimension (enforced with max_len = self.num_imgs)
        time_pos_encoding = PositionalEncoding(
                batch_size=B,
                num_slots=num_slots,
                d_model=self.token_dim,
                max_len=num_imgs
            ).to(inputs.device)
        time_encoded_input = time_pos_encoding(token_input).reshape(B, num_imgs * num_slots, self.token_dim)

        # feeding through transformer encoder blocks
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output, src_mask=mask)
        token_output = token_output.reshape(B, num_imgs, num_slots, self.token_dim)

        # mapping back to slot dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output

    def get_mask_pattern(self, seq_len, num_slots):
        """
        Obtaining a binary maskign pattern to avoid attending to future time steps
        """
        num_tokens = seq_len * num_slots
        mask_pattern = torch.zeros((num_tokens, num_tokens))
        for i in range(seq_len):
            mask_pattern[num_slots*i:, num_slots*i:num_slots*(i + 1)] = 1.
        mask = mask_pattern.clone().float()
        mask = mask.masked_fill(mask_pattern == 0, float('-inf'))
        mask = mask.masked_fill(mask_pattern == 1, float(0.0))
        return mask, mask_pattern


# TODO: IT could be improved by caching the encoding, and repeating for batch and slot in forward pass
class PositionalEncoding(nn.Module):
    """
    Positional encoding to be added to the input tokens of the transofrmer predictor.

    Our positional encoding only informs about the time-step, i.e., all slots extracted
    from the same input frame share the same positional embedding. This allows our predictor
    model to maintain the permutation equivariance properties.

    Args:
    -----
    batch_size: int
        Number of elements in the batch.
    num_slots: int
        Number of slots extracted per frame. Positional encoding will be repeat for each of these.
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Percentage of dropout to apply after adding the poisitional encoding. Default is 0.1
    max_len: int
        Length of the sequence.
    """

    def __init__(self, batch_size, num_slots, d_model, dropout=0.1, max_len=5000):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # initializing sinusoidal positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, 1, d_model)

        pe = pe.repeat(batch_size, 1, num_slots, 1)  # NOTE: this is input-dependent
        self.register_buffer('pe', pe)  # Shape is (B, Max_Len, num_slots, d_model)
        return

    def forward(self, x: torch.tensor):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding. Shape is (B, Seq_len, Num_Slots, Token_Dim)
        """
        if x.shape[1] != self.pe.shape[1]:
            raise ValueError(f"Seq length {x.shape[1]} does not match PE-length {self.pe.shape[1]}...")
        x = x + self.pe
        y = self.dropout(x)
        return y


class OCVTransformerV1Predictor(nn.Module):
    """
    Version 1 of our Object-Centric Transformer Predictor Module.
    This module models the object motion and object interactions in a dissentangled manner by
    sequentially applying object- and time-attention, i.e. [time, obj, time, ...]

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False):
        """ Module Initialzer """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating Object-Centric Transformer-v1 Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[ObjectCentricTransformerLayerV1(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads
                ) for _ in range(num_layers)]
            )
        return

    def get_attn_maps(self, inputs):
        """
        Obtaining the attention maps from the last transformer layer
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # obtaining mask-patterns for both the object- and time-attention mechanisms
        time_mask, _ = self.get_time_mask_pattern(num_imgs)
        time_mask = time_mask.to(inputs.device)

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_positional_encoding = PositionalEncoding(
                batch_size=B,
                num_slots=num_slots,
                d_model=self.token_dim,
                max_len=num_imgs
            ).to(inputs.device)
        time_encoded_input = time_positional_encoding(token_input)

        # feeding through transformer encoder blocks without last layer
        token_output = time_encoded_input
        for encoder in self.transformer_encoders[:-1]:
            token_output = encoder(token_output, time_mask=time_mask)
        last_encoder_layer = self.transformer_encoders[-1]

        # get object block attention map
        token_output = token_output.reshape(B * num_imgs, num_slots, slot_dim)
        object_norm_output = last_encoder_layer.object_encoder_block.norm1(token_output)
        _, object_attn_weights = last_encoder_layer.object_encoder_block.self_attn(
                object_norm_output,
                object_norm_output,
                object_norm_output,
                key_padding_mask=None,
                need_weights=True
            )
        object_attn_weights = object_attn_weights.view(B, num_imgs, num_slots, num_slots)

        # get time block attention map
        token_output = token_output.view(B * num_imgs, num_slots, slot_dim)
        object_encoder_block_out = last_encoder_layer.object_encoder_block(token_output)
        object_encoder_block_out = object_encoder_block_out.view(B, num_imgs, num_slots, slot_dim)
        object_encoder_block_out = object_encoder_block_out.transpose(1, 2)
        object_encoder_block_out = object_encoder_block_out.reshape(B * num_slots, num_imgs, slot_dim)
        time_norm_output = last_encoder_layer.time_encoder_block.norm1(object_encoder_block_out)
        _, time_attn_weights = last_encoder_layer.time_encoder_block.self_attn(
                time_norm_output,
                time_norm_output,
                time_norm_output,
                attn_mask=time_mask,
                key_padding_mask=None,
                need_weights=True
            )
        time_attn_weights = time_attn_weights.view(B, num_slots, num_imgs, num_imgs)

        return object_attn_weights, time_attn_weights

    def forward(self, inputs):
        """ Forward pass """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # obtaining mask-patterns for both the object- and time-attention mechanisms
        time_mask, _ = self.get_time_mask_pattern(num_imgs)
        time_mask = time_mask.to(inputs.device)

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_positional_encoding = PositionalEncoding(
                batch_size=B,
                num_slots=num_slots,
                d_model=self.token_dim,
                max_len=num_imgs
            ).to(inputs.device)
        time_encoded_input = time_positional_encoding(token_input)  # (B, num_imgs, num_slots, token_dim)
        token_output = time_encoded_input

        # feeding through transformer encoder blocks
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output, time_mask=time_mask)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output

    def get_time_mask_pattern(self, seq_len):
        """ Creating a mask to only attent to the same object during past time-steps """
        mask_pattern = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1).type(torch.uint8)
        mask = mask_pattern.clone().float()
        mask = mask.masked_fill(mask_pattern == 0, float('-inf'))
        mask = mask.masked_fill(mask_pattern == 1, float(0.0))
        return mask, mask_pattern


class ObjectCentricTransformerLayerV1(nn.Module):
    """
    Object-Centric Transformer-v1 Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """ Module initializer """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.nhead = n_heads

        self.object_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        return

    def forward(self, inputs, time_mask):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out, src_mask=time_mask)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out


class OCVTransformerV2Predictor(nn.Module):
    """
    Version 2 of our Object-Centric Transformer Predictor Module.
    This module models the object motion and object interactions in a dissentangled manner by
    applying object- and time-attention in parallel.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    """

    def __init__(self, num_slots, slot_dim, num_imgs, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False):
        """ Module initializer """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print_("Instanciating Object-Centric Transformer-v2 Predictor:")
        print_(f"  --> num_layers: {self.num_layers}")
        print_(f"  --> input_dim: {self.slot_dim}")
        print_(f"  --> token_dim: {self.token_dim}")
        print_(f"  --> hidden_dim: {self.hidden_dim}")
        print_(f"  --> num_heads: {self.nhead}")
        print_(f"  --> residual: {self.residual}")
        print_("  --> batch_first: True")
        print_("  --> norm_first: True")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim will be split across num_heads, i.e. each head will have dim embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[ObjectCentricTransformerLayerV2(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )
        return

    def get_attn_maps(self, inputs):
        """
        Obtaining the attention maps from the last transformer layer
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # obtaining mask-patterns for both the object- and time-attention mechanisms
        time_mask, _ = self.get_time_mask_pattern(num_imgs)
        time_mask = time_mask.to(inputs.device)

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_positional_encoding = PositionalEncoding(
                batch_size=B,
                num_slots=num_slots,
                d_model=self.token_dim,
                max_len=num_imgs
            ).to(inputs.device)
        time_encoded_input = time_positional_encoding(token_input)
        token_output = time_encoded_input

        # feeding tokens through transformer la<ers without last layer
        for encoder in self.transformer_encoders[:-1]:
            token_output = encoder(token_output, time_mask=time_mask)
        last_encoder_layer = self.transformer_encoders[-1]
        norm_output = last_encoder_layer.norm1(token_output)

        norm_output_aux = norm_output.clone().reshape(B * num_imgs, num_slots, slot_dim)
        _, object_attn_weights = last_encoder_layer.self_attn_obj(
                norm_output_aux,
                norm_output_aux,
                norm_output_aux,
                key_padding_mask=None,
                need_weights=True
            )
        object_attn_weights = object_attn_weights.reshape(B, num_imgs, num_slots, num_slots)

        norm_output = norm_output.transpose(1, 2).reshape(B * num_slots, num_imgs, slot_dim)
        _, time_attn_weights = last_encoder_layer.self_attn_time(
                norm_output,
                norm_output,
                norm_output,
                attn_mask=time_mask,
                key_padding_mask=None,
                need_weights=True
            )
        time_attn_weights = time_attn_weights.reshape(B, num_slots, num_imgs, num_imgs)

        return object_attn_weights, time_attn_weights

    def forward(self, inputs):
        """ Forward pass """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # obtaining mask-patterns for both the object- and time-attention mechanisms
        time_mask, _ = self.get_time_mask_pattern(num_imgs)
        time_mask = time_mask.to(inputs.device)

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_positional_encoding = PositionalEncoding(
                batch_size=B,
                num_slots=num_slots,
                d_model=self.token_dim,
                max_len=num_imgs
            ).to(inputs.device)
        time_encoded_input = time_positional_encoding(token_input)

        # feeding tokens through transformer la<ers
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output, time_mask=time_mask)

        # projecting back to slot-dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output

    def get_time_mask_pattern(self, seq_len):
        """ Creating a mask to only attent to the same object during past time-steps """
        mask_pattern = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1).type(torch.uint8)
        mask = mask_pattern.clone().float()
        mask = mask.masked_fill(mask_pattern == 0, float('-inf'))
        mask = mask.masked_fill(mask_pattern == 1, float(0.0))
        return mask, mask_pattern


class ObjectCentricTransformerLayerV2(nn.TransformerEncoderLayer):
    """
    Version 2 of our Object-Centric Transformer Module.
    This module models the object motion and object interactions in a dissentangled manner by
    applying object- and time-attention in parallel.

    Args:
    -----
    TODO: Too lazy for this now
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        """ Module initializer """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=dtype
            )

        self.self_attn_obj = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        self.self_attn_time = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        return

    def forward(self, src, time_mask):
        """
        Forward pass through the Object-Centric Transformer-v2.
        Overloads PyTorch's transformer forward pass.

        Args:
        -----
        src: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), time_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, time_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, time_mask):
        """ Forward pass through the parallel attention branches """
        B, num_imgs, num_slots, dim = x.shape

        # object-attention
        x_aux = x.clone().view(B * num_imgs, num_slots, dim)
        x_obj = self.self_attn_obj(
                query=x_aux,
                key=x_aux,
                value=x_aux,
                need_weights=False
            )[0]
        x_obj = x_obj.view(B, num_imgs, num_slots, dim)

        # time-attention
        x = x.transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
        x_time = self.self_attn_time(
                query=x,
                key=x,
                value=x,
                attn_mask=time_mask,
                need_weights=False
            )[0]
        x_time = x_time.view(B, num_slots, num_imgs, dim).transpose(1, 2)

        # NOTE: IF 'need_weights' is False, is the indexing [0] needed?
        y = self.dropout1(x_obj + x_time)
        return y


class MLPMixer(nn.Module):
    """
    MLP-Mixer based predictor module.
    This module sequentially applies MLP-Mixer Layers, which first apply an MLP over the objects,
    and then apply an MLP over the time dimension.
      --> Inspired by https://github.com/MotionMLP/MotionMixer/blob/main/h36m/mlp_mixer.py

    Args:
    -----
    seq_length: int
        Number of images/frames to jointly process
    num_objs: int
        Number of object slots per image
    slot_dim: int
        Dimensionality of the object slots
    token_dim: int
        Object slots are embedded into this dimensionality via a Fully Connected layer
    hidden_dim: int
        Hidden dimensionality of the the MLP modules
    num_layers: int
        Number of MLP-Blocks to sequentially apply
    dropout: float
        Amount of dropout to apply after each FC layer in the MLP blocks
    object_mixing: bool
        If True, the MLP-Object-Mixer is applied to jointly process all slots in the same image
    temporal_mixing: bool
        If True, the MLP-Time-Mixer is applied to jointly process a slot on all previous time steps
    se_weighting: bool
        If True, a Squeeze-and-Excitation block is applied after each MLP-Block to recalibrate
        the output values. This idea is borrowed from the MotionMixer paper.
    """

    def __init__(self, seq_length, num_objs, slot_dim, token_dim, hidden_dim, num_layers, dropout=0.,
                 object_mixing=True, temporal_mixing=True, se_weighting=True, **kwargs):
        """ Module initializer """
        super().__init__()
        self.seq_length = seq_length
        self.num_objs = num_objs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.object_mixing = object_mixing
        self.temporal_mixing = temporal_mixing
        self.se_weighting = se_weighting

        print_("Instanciating MLP-Mixer Predictor:")
        print_(f"  --> num_layers:      {self.num_layers}")
        print_(f"  --> num_objs:        {self.num_objs}")
        print_(f"  --> seq_length:      {self.seq_length}")
        print_(f"  --> input_dim:       {self.slot_dim}")
        print_(f"  --> token_dim:       {self.token_dim}")
        print_(f"  --> hidden_dim:      {self.hidden_dim}")
        print_(f"  --> dropout:         {self.dropout}")
        print_(f"  --> object_mixing:   {self.object_mixing}")
        print_(f"  --> temporal_mixing: {self.temporal_mixing}")
        print_(f"  --> se_weighting:    {self.se_weighting}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        self.mlp_mixer_blocks = nn.ModuleList(
                MixerBlock(
                    num_objs=num_objs,
                    seq_len=seq_length,
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    object_mixing=object_mixing,
                    temporal_mixing=temporal_mixing,
                    se_weighting=se_weighting,
                    **kwargs
                ) for _ in range(num_layers)
            )

        return

    def forward(self, x):
        """
        Forward pass.

        Args:
        -----
        x: torch Tensor
            Slots from the input images. Shape is (B, Num_Imgs, Num_Slots, Slot_dim)
        """
        mlp_in = self.mlp_in(x)
        for mlp_block in self.mlp_mixer_blocks:
            mlp_in = mlp_block(mlp_in)
        y = self.mlp_out(mlp_in)
        return y


class MixerBlock(nn.Module):
    """
    MLP-Mixer Block.

    Sequentially applies two residula MLPs, iterleaved with Layer-Normalization:
        - The fist MLP simultaneously processes objects from at the same time step
        - The second MLP processes an object over all previous time steps.

    Args:
    -----
    num_objs: int
        Number of object slots per image
    seq_length: int
        Number of images/frames to jointly process
    token_dim: int
        Object slots are embedded into this dimensionality via a Fully Connected layer
    hidden_dim: int
        Hidden dimensionality of the the MLP modules
    dropout: float
        Amount of dropout to apply after each FC layer in the MLP blocks
    object_mixing: bool
        If True, the MLP-Object-Mixer is applied to jointly process all slots in the same image
    temporal_mixing: bool
        If True, the MLP-Time-Mixer is applied to jointly process a slot on all previous time steps
    se_weighting: bool
        If True, a Squeeze-and-Excitation block is applied after each MLP-Block to recalibrate
        the output values. This idea is borrowed from the MotionMixer paper.
    """

    def __init__(self, num_objs, seq_len, token_dim, hidden_dim, dropout=0., object_mixing=True,
                 temporal_mixing=True, se_weighting=True, **kwargs):
        """ Module initializer """
        super().__init__()
        self.num_objs = num_objs
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.object_mixing = object_mixing
        self.temporal_mixing = temporal_mixing
        self.se_weighting = se_weighting
        self.se_feature_ratio = kwargs.get("se_feature_ratio", 4)

        # object mixing modules
        if object_mixing:
            self.slot_mlp = MLPBlock(
                    input_dim=num_objs * token_dim,
                    hidden_dim=hidden_dim,
                    output_dim=num_objs * token_dim,
                    dropout=dropout
                )
        if se_weighting and object_mixing:
            self.se_net_object = SEBlock(input_dim=seq_len)

        # temporal mixing blocks
        if temporal_mixing:
            self.time_mlp = MLPBlock(
                    input_dim=seq_len * token_dim,
                    hidden_dim=hidden_dim,
                    output_dim=seq_len * token_dim,
                    dropout=dropout
                )
        if se_weighting and object_mixing:
            self.se_net_time = SEBlock(input_dim=num_objs)

        # normalizations
        self.layer_norm_object = nn.LayerNorm(token_dim * num_objs)
        self.layer_norm_time = nn.LayerNorm(token_dim * seq_len)
        return

    def forward(self, x):
        """
        Forward pass.

        Args:
        -----
        x: torch Tensor
            Input tokens to process with the mixer-block. Shape is (B, num_imgs, num_slots, token_dim)
        """
        B, num_imgs, num_slots, token_dim = x.shape

        # object mixing
        if self.object_mixing:
            x = x.view(B, num_imgs, num_slots * token_dim)  # (Batch, time, objs)
            y = self.layer_norm_object(x)
            y = self.slot_mlp(y)
            if self.se_weighting:
                y = self.se_net_object(y)
            x = x + y
            x = x.view(B, num_imgs, num_slots, token_dim)

        # temporal mixing
        if self.temporal_mixing:
            x = x.transpose(1, 2).contiguous().view(B, num_slots, num_imgs * token_dim)  # (B, objs, time)
            y = self.layer_norm_time(x)
            y = self.time_mlp(y)
            if self.se_weighting:
                y = self.se_net_time(y)
            x = x + y
            x = y.view(B, num_slots, num_imgs, token_dim).transpose(1, 2).contiguous()

        return x


#
