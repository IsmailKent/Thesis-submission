"""
Accesing models
"""

from .encoders_decoders import get_encoder, get_decoder, SoftPositionEmbed, InstanceMaskEncoder,\
    SimpleConvEncoder, DownsamplingConvEncoder, InstanceEncoderModule
from .initializers import get_initalizer

from .attention import setup_attention, SlotAttention, MultiHeadSelfAttention, DotProdCrossAttention
from .TransformerBlock import TransformerBlock
from .SlotAttentionModel import SlotAttentionModel
from .SAVi import SAVi
from .RecursiveInstanceEncoder import RecursiveInstanceEncoder
from .Predictors import LSTMPredictor
from .model_utils import freeze_params
