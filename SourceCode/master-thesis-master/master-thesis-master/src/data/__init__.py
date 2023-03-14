"""
Accessing datasets from script
"""

# from ._Movi import _MoviA, _MoviC
from .Movi import MOVI
from .moving_mnist import CustomMovingMNIST
from .MultiDSprites import MultiDSprites
from .obj3d import OBJ3D
from .PhysicalConcepts import PhysicalConcepts
from .Sketchy import Sketchy
from .SpritesMOT import SpritesDataset
from .SynpickVP import SynpickVP, SynpickInstances
from .tetrominoes import Tetrominoes

from .load_data import load_data, build_data_loader
