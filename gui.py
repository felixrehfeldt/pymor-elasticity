from __future__ import division, absolute_import, print_function

import numpy as np

from pymor.la import NumpyVectorArray
from pymor.la.interfaces import VectorArrayInterface
from pymor.grids.referenceelements import triangle
from pymor.core.cache import cached
from pymor.core import BasicInterface


class DisplacementVisualizer(BasicInterface):

    def __init__(self, grid, bounding_box=[[0, 0], [1, 1]], backend=None, block=False):
        assert grid.reference_element is triangle
        self.grid = grid
        self.bounding_box = bounding_box
        self.backend = backend
        self.block = block

    def visualize(self, U, discretization, title=None, legend=None, separate_colorbars=False,
                  block=None):
        assert isinstance(U, VectorArrayInterface)
