# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import ImmutableInterface
from pymor.domaindescriptions import RectDomain
from pymor.functions import ConstantFunction
from pymor.tools import Named


class ElasticityProblem(ImmutableInterface, Named):

    def __init__(self, domain=RectDomain(), rhs=ConstantFunction(dim_domain=2),
                 volume_force=ConstantFunction(np.array([0., -1.]), dim_domain=2),
                 name=None):
        self.domain = domain
        self.rhs = rhs
        self.volume_force = volume_force
        self.name = name
