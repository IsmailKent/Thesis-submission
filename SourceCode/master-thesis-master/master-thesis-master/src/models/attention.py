"""
Attention modules:
"""

import torch
import torch.nn as nn


ATTENTION = ["DotProdSelf", "DotProdCross", "MultiHeadSelf", "SlotAttention", "CustomSlotAttention"]


def setup_attention(attention_type, **kwargs):
    """
    Loading an attention module
    """
    if attention_type not in ATTENTION:
        raise ValueError(f"Unknown attention type {attention_type}. Use one of {ATTENTION}")

    if(attention_type == "DotProdSelf"):
        attention = DotProdSelfAttention(**kwargs)
    elif(attention_type == "DotProdCross"):
        attention = DotProdCrossAttention(
                emb_dim=kwargs["emb_dim"]
            )
    elif(attention_type == "MultiHeadSelf"):
        num_heads = getattr(kwargs, "num_heads", 8)
        attention = MultiHeadSelfAttention(
                emb_dim=kwargs["emb_dim"],
                num_heads=num_heads
            )
    elif(attention_type == "SlotAttention"):
        attention = SlotAttention(
                dim_feats=kwargs["emb_dim"],
                num_iters=kwargs["num_iters"],
                num_slots=kwargs["num_slots"]
            )
    elif(attention_type == "CustomSlotAttention"):
        attention = CustomSlotAttention(
                dim_feats=kwargs["emb_dim"],
                num_iters=kwargs["num_iters"],
                num_slots=kwargs["num_slots"]
            )
    else:
        raise NotImplementedError(f"Unknown attention type. Use one of {ATTENTION}")

    return attention


class CustomSlotAttention(nn.Module):
    """
    Slight modification of the Slot Attention module from:
      Locatello, Francesco, et al. "Object-centric learning with slot attention." NeurIPS 2020
    """

    def __init__(self, dim_emb, dim_feats, dim_slots, epsilon=1e-8):
        """ Module Initializer """
        super().__init__()
        self.scale = dim_emb ** -0.5
        self.epsilon = epsilon

        # normalization layers
        self.norm_input = nn.LayerNorm(dim_feats)
        self.norm_slot = nn.LayerNorm(dim_slots)

        # attention embedders
        self.to_q = nn.Linear(dim_slots, dim_emb)
        self.to_k = nn.Linear(dim_feats, dim_emb)
        self.to_v = nn.Linear(dim_feats, dim_emb)
        self.out_proj = nn.Linear(dim_emb, dim_slots)
        return

    def forward(self, slots, feats):
        """
        Forward pass through the Modified Slot Attention mechanism

        Args:
        -----
        slots: torch Tensor
            Current state of the slot represetations.
            Shape is (Batch, Num Slots, Slot Dimensionality)
        feats: torch Tensor
            input feature vectors extracted by the encoder.
            Shape is (Batch, Num Locations, Feature Dimensionality)

        Returns:
        --------
        slots: torch Tensor
            refined input slots
            Shape is (Batch, Num Slots, Slot Dimensionality)
        """
        _, num_slots, dim_slots = slots.shape
        B, num_feats, dim_feats = feats.shape
        self.attention_masks = None

        # normalization and projection
        feats = self.norm_input(feats)
        k, v = self.to_k(feats), self.to_v(feats)
        slots = self.norm_slot(slots)
        q = self.to_q(slots)

        # slot-attention
        dots = torch.einsum('b i d , b j d -> b i j', q, k) / self.scale
        attn = dots.softmax(dim=1) + self.epsilon
        self.attention_masks = attn
        slots = torch.einsum('b i d , b d j -> b i j', attn, v)

        slots = self.out_proj(slots)
        return slots


class SlotAttention(nn.Module):
    """
    Implementation of the SlotAttention module from:
      --> Locatello, Francesco, et al. "Object-centric learning with slot attention." NeurIPS 2020

    Args:
    -----
    dim_feats: integer
        dimensionality of the input embeddings
    dim_slots: integer
        dimensionality of the slots
    num_slots: integer
        number of slots competing for the represetnations
    num_iters: integer
        nubmer of recurrent iterations to refine the slots
    mlp_hidden_size: integer
        hidden dimensionality of the mlp,
    """

    def __init__(self, dim_feats, dim_slots, num_slots, num_iters_first=2, num_iters=2,
                 mlp_hidden=128, epsilon=1e-8):
        """ Module Initializer """
        super().__init__()
        self.dim_slots = dim_slots
        self.num_iters_first = num_iters_first
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.epsilon = epsilon
        self.scale = dim_feats ** -0.5

        # normalization layers
        self.norm_input = nn.LayerNorm(dim_feats, eps=0.001)
        self.norm_slot = nn.LayerNorm(dim_slots, eps=0.001)
        self.norm_mlp = nn.LayerNorm(dim_slots, eps=0.001)

        # attention embedders
        self.to_q = nn.Linear(dim_slots, dim_slots)
        self.to_k = nn.Linear(dim_feats, dim_slots)
        self.to_v = nn.Linear(dim_feats, dim_slots)

        # Slot update functions.
        self.gru = nn.GRUCell(dim_slots, dim_slots)
        self.mlp = nn.Sequential(
            nn.Linear(dim_slots, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, dim_slots),
        )
        return

    def forward(self, inputs, slots, step=0, **kwargs):
        """
        Forward pass as depicted in Algorithm 1 from paper

        Args:
        -----
        inputs: torch Tensor
            input feature vectors extracted by the encoder.
            Shape is (Batch, Num locations, Dimensionality)

        Returns:
        --------
        slots: torch Tensor
            Slot assignment for each of the input vectors
            Shape is (Batch, Num Slots, Slot Dimensionality)
        """
        B, N, D = inputs.shape
        self.attention_masks = None

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        # iterative refinement of the slot representation
        num_iters = self.num_iters_first if step == 0 else self.num_iters
        for _ in range(num_iters):
            slots_prev = slots
            slots = self.norm_slot(slots)
            q = self.to_q(slots)

            # q ~ (B, N_Slots, Slot_dim)
            # k, v ~ (B, N_locs, Slot_dim)
            # attention equation [softmax(Q K^T) V]
            dots = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale  # dots ~ (B, N_slots, N_locs)
            attn = dots.softmax(dim=1) + self.epsilon  # enforcing competition between slots
            attn = attn / attn.sum(dim=-1, keepdim=True)  # attn ~ (B, N_slots, N_locs)
            self.attention_masks = attn
            updates = torch.einsum('b i d , b d j -> b i j', attn, v)  # updates ~ (B, N_slots, slot_dim)
            # further refinement
            slots = self.gru(
                updates.reshape(-1, self.dim_slots),
                slots_prev.reshape(-1, self.dim_slots)
            )
            slots = slots.reshape(B, -1, self.dim_slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def get_attention_masks(self, reshape=None):
        """
        Fetching last computer attention masks

        Args:
        -----
        reshape: list/tuple
            If not None, masks are reshaped to this spatial dimensions

        Returns:
        --------
        attention_masks: torch Tensor
            attention masks highligtinh the importance of each location to each slot
            Shape is (B, N_slots, N_locs)
        """
        B, N_slots, N_locs = self.attention_masks.shape

        if(reshape):
            masks = self.attention_masks.reshape(B, N_slots, *reshape)
        else:
            masks = self.attention_masks

        return masks


class MetaAttention(nn.Module):
    """
    MetaClass for (Multi-Head) Key-Value Attention Mechanisms

    Args:
    -----
    emb_dim: integer
        dimensionality of the token embeddings. In our particular case it
        corresponds to the dimensionality of projected linearized patches
    num_heads: integer
        number of heads accross which we compute attention.
        Head_dim = Emb_dim / Num_Heads. Division must be exact.
    dropout: float [0,1]
        Percentage of connections dropped during the attention
    """

    def __init__(self, emb_dim, num_heads=1, dropout=0., out_dim=None, **kwargs):
        """
        Initializer of the attention block
        """
        assert emb_dim % num_heads == 0, "Embedding dimension needs to be divisible by number of heads..."

        super(MetaAttention, self).__init__()
        out_dim = out_dim if out_dim is not None else emb_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        # computing query, key, value for all embedding heads
        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        # output projection
        self.out_projection = nn.Sequential(
            nn.Linear(emb_dim, out_dim, bias=False),
        )

        self.attention_masks = None

        return

    def forward(self, x):
        """ """
        raise NotImplementedError("Meta-Class does not implement a 'forward' method...")

    def attention(self, query, key, value, dim_head):
        """ Implementation of the standard attention equation """
        scale = dim_head ** -0.5  # 1/sqrt(d_k)
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale  # Q * K.T / sqrt(d_k)
        attention = dots.softmax(dim=-1)
        # attention = attention / attention.sum(dim=-1, keepdim=True)  # attn ~ (B, N_slots, N_locs)
        self.attention_masks = attention
        attention = self.drop(attention)
        vect = torch.einsum('b i d , b d j -> b i j', attention, value)  # Att * V

        return vect

    def get_attention_masks(self, reshape=None):
        """
        Fetching last computer attention masks

        Args:
        -----
        reshape: list/tuple
            If not None, masks are reshaped to this spatial dimensions

        Returns:
        --------
        attention_masks: torch Tensor
            attention masks highlighting the importance of each location to each slot
            Shape is (B, N_tokens, Token_dim)
        """
        assert self.attention_masks is not None, "Attention masks are none..."
        B, N_tokens, Token_dim = self.attention_masks.shape

        if(reshape):
            masks = self.attention_masks.reshape(B, N_tokens, *reshape)
        else:
            masks = self.attention_masks

        return masks


class DotProdSelfAttention(MetaAttention):
    """
    Vanilla Key-Value self-attention mechanism through dot-product using only one attention head.
    Key, value and query come from the same input
    """

    def __init__(self, emb_dim, dropout=0., **kwargs):
        """ Initializer of the attention block """
        super().__init__(
                emb_dim=emb_dim,
                dropout=dropout,
                **kwargs
            )
        return

    def forward(self, x, **kwargs):
        """ Forward pass through multi-head attention """
        batch_size, num_tokens, token_dim = x.size()
        # linear projections
        q, k, v = self.q(x), self.k(x), self.v(x)
        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=token_dim)
        # output projection
        y = self.out_projection(vect)
        return y


class DotProdCrossAttention(MetaAttention):
    """
    Vanilla Key-Value attention mechanism through dot-product using only one attention head
    Key and value come from the same input. Query is different
    """

    def __init__(self, emb_dim, dropout=0.):
        """ Initializer of the attention block """
        super().__init__(
                emb_dim=emb_dim,
                dropout=dropout
            )
        return

    def forward(self, x, query_embs, **kwargs):
        """ Forward pass through multi-head attention """
        batch_size, num_tokens, token_dim = x.size()
        # linear projections
        q = self.q(query_embs)
        k, v = self.k(x), self.v(x)
        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=token_dim)
        # output projection
        y = self.out_projection(vect)
        return y


class MultiHeadSelfAttention(MetaAttention):
    """ Vanilla multi-head dot-product attention mechanism """

    def __init__(self, emb_dim, num_heads=8, dropout=0.):
        """ Initializer of the attention block """
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        return

    def forward(self, x, **kwargs):
        """
        Forward pass through multi-head attention
        """

        batch_size, num_tokens, token_dim = x.size()
        dim_head = token_dim // self.num_heads

        # linear projections
        q, k, v = self.q(x), self.k(x), self.v(x)

        # split into heads and move to batch-size side: (Batch, Token, Dims) --> (Batch, Heads, Token, HeadDim) --> (Batch* Heads, Token, HeadDim)
        q = q.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        q = q.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        k = k.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        k = k.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        v = v.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        v = v.reshape(batch_size * self.num_heads, num_tokens, dim_head)

        # applying attention equation
        vect = self.attention(query=q, key=k, value=v, dim_head=dim_head)  # shape (Batch* Heads, Token, HeadDim)
        vect = vect.reshape(batch_size, self.num_heads, num_tokens, dim_head).transpose(1, 2)  # back to original shape now
        # rearranging heads and recovering original shape
        vect = vect.reshape(batch_size * num_tokens, self.num_heads * dim_head)  # TODO contiguous()?

        # output projection
        y = self.out_projection(vect)

        return y.reshape(batch_size, num_tokens, self.num_heads*dim_head)


# TODO: Missing Multi-Head Cross-Attention


#
