from __future__ import absolute_import, division, print_function

import numpy as np
import sys

from PySide import QtGui, QtCore
from docopt import docopt

from pymor import defaults
defaults.default_sparse_solver = 'spsolve'

from pymor.domaindescriptions import RectDomain, BoundaryType
from pymor.gui.glumpy import GlumpyPatchWidget

from problem import ElasticityProblem
from discretizer import discretize_elasticity_cg

if len(sys.argv) < 4:
    print('Usage: demo.py BOUNDARIES LAMBDA MU')
    sys.exit(-1)


# parse BOUNDARIES argument
boundaries = sys.argv[1]
assert len(boundaries) == 4
boundaries = int(boundaries)
boundary_types = []
for _ in xrange(4):
    b = boundaries % 10
    assert b in {0, 1}
    boundary_types.append(BoundaryType('dirichlet') if b else None)
    boundaries //= 10
boundary_types = dict(zip(('top', 'right', 'bottom', 'left'), boundary_types))

domain = RectDomain([[0., 0.], [3., 1.]], **boundary_types)
p = ElasticityProblem(domain)
d, d_data = discretize_elasticity_cg(p, diameter=np.sqrt(2)/10)

lambda_ = float(sys.argv[2])
mu = float(sys.argv[3])
U = d.solve((lambda_, mu))

d.visualize(U)
