# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.discretizations import StationaryDiscretization
from pymor.domaindiscretizers import discretize_domain_default
from pymor.grids import TriaGrid

from operators import (ElasticityOperator, material_elasticity_tensor, VolumeForceFunctional,
                       H1ProductOperator)
from gui import DisplacementVisualizer
from pymor.parameters.functionals import ProjectionParameterFunctional


def discretize_elasticity_cg(analytical_problem, diameter=None, domain_discretizer=None,
                             grid=None, boundary_info=None):

    assert grid is None or boundary_info is not None
    assert boundary_info is None or grid is not None
    assert grid is None or domain_discretizer is None

    p = analytical_problem

    if grid is None:
        domain_discretizer = domain_discretizer or discretize_domain_default
        if diameter is None:
            grid, boundary_info = domain_discretizer(analytical_problem.domain)
        else:
            grid, boundary_info = domain_discretizer(analytical_problem.domain, diameter=diameter)

    assert isinstance(grid, TriaGrid)

    Operator = ElasticityOperator
    # Functional = None

    C = material_elasticity_tensor(0., 0.)
    L_0 = ElasticityOperator(grid, boundary_info, C, name='affine_part')
    C = material_elasticity_tensor(mu=1., lambda_=0.)
    L_mu = ElasticityOperator(grid, boundary_info, C, name='mu_part', dirichlet_clear_diag=True)
    C = material_elasticity_tensor(mu=0., lambda_=1.)
    L_lambda = ElasticityOperator(grid, boundary_info, C, name='lambda_part', dirichlet_clear_diag=True)

    theta_mu = ProjectionParameterFunctional('mu', tuple())
    theta_lambda = ProjectionParameterFunctional('lambda', tuple())

    L = L_0.lincomb([L_0, L_mu, L_lambda], coefficients=[1., theta_mu, theta_lambda])

    F = VolumeForceFunctional(grid, p.volume_force, boundary_info=boundary_info)

    visualizer = DisplacementVisualizer(grid=grid, bounding_box=[[-1., -1.], [4., 2.]])

    products = {'h1': H1ProductOperator(grid, boundary_info)}

    parameter_space = p.parameter_space if hasattr(p, 'parameter_space') else None

    discretization = StationaryDiscretization(L, F, products=products, visualizer=visualizer,
                                              parameter_space=parameter_space, name='{}_CG'.format(p.name))

    return discretization, {'grid': grid, 'boundary_info': boundary_info}
