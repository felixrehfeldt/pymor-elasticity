from __future__ import division, absolute_import, print_function

import numpy as np
from itertools import product

from pymor.la import NumpyVectorArray
from pymor.grids.referenceelements import triangle
from pymor.operators.basic import NumpyMatrixBasedOperator, NumpyMatrixOperator
from scipy.sparse import coo_matrix, csc_matrix


shape_functions_1d = \
    [lambda X: X[..., 0] * (-1.) + X[..., 1] * (-1.) + 1.,
     lambda X: X[..., 0],
     lambda X: X[..., 1]]

def shape_function_factory_2d(shape_function_1d, dim):

    def sf(X):
        R = np.zeros(X.shape)
        R[..., dim] = shape_function_1d(X)
        return R

    return sf

shape_functions = [shape_function_factory_2d(sf, dim) for dim, sf in product(range(2), shape_functions_1d)]

shape_function_gradients_1d = \
    np.array([[-1., -1.],
              [1., 0.],
              [0., 1.]])

shape_function_gradients = np.zeros((2 * len(shape_function_gradients_1d), 2, 2))
for i, (dim, sfg) in enumerate(product(range(2), shape_function_gradients_1d)):
    shape_function_gradients[i, dim] = sfg


class VolumeForceFunctional(NumpyMatrixBasedOperator):

    sparse = False

    def __init__(self, grid, function, boundary_info=None, dirichlet_data=None, order=2, name=None):
        assert grid.reference_element(0) is triangle
        assert function.shape_range == (2,)
        self.dim_source = grid.size(grid.dim) * 2
        self.dim_range = 1
        self.grid = grid
        self.boundary_info = boundary_info
        self.function = function
        self.dirichlet_data = dirichlet_data
        self.order = order
        self.name = name
        self.build_parameter_type(inherits=(function, dirichlet_data))

    def _assemble(self, mu=None):
        mu = self.parse_parameter(mu)
        g = self.grid
        bi = self.boundary_info

        # evaluate function at all quadrature points -> shape = (g.size(0), number of quadrature points)
        F = self.function(g.quadrature_points(0, order=self.order), mu=mu)

        # evaluate the shape functions at the quadrature points on the reference
        # element -> shape = (number of shape functions, number of quadrature points)
        q, w = g.reference_element.quadrature(order=self.order)

        # integrate the products of the function with the shape functions on each element
        # -> shape = (g.size(0), number of shape functions)
        SF = np.array([f(q) for f in shape_functions])
        SF_INTS = np.einsum('eid,pid,e,i->ep', F, SF, g.integration_elements(0), w).ravel()

        # map local DOFs to global DOFS
        # FIXME This implementation is horrible, find a better way!
        SF_I = np.concatenate((g.subentities(0, 2) * 2, g.subentities(0, 2) * 2 + 1), 1).ravel()
        I = np.array(coo_matrix((SF_INTS, (np.zeros_like(SF_I), SF_I)), shape=(1, g.size(g.dim) * 2)).todense()).ravel()

        # boundary treatment
        if bi is not None and bi.has_dirichlet:
            DI = bi.dirichlet_boundaries(g.dim)
            if self.dirichlet_data is not None:
                D = self.dirichlet_data(g.centers(g.dim)[DI], mu=mu)
                I[DI * 2] = D[:, 0]
                I[DI * 2 + 1] = D[:, 1]
            else:
                I[DI * 2] = 0
                I[DI * 2 + 1] = 0

        return NumpyMatrixOperator(I.reshape((1, -1)))


def material_elasticity_tensor(mu, lambda_, dim=2):

    def dirac(i, j):
        return 1 if i == j else 0

    C = np.zeros((dim,) * 4, dtype=np.float)
    for i, j, k, l in product(xrange(dim), repeat=4):
        C[i, j, k, l] = (lambda_ * dirac(i, j) * dirac(k, l) +
                         mu * (dirac(i, k) * dirac(j, l) + dirac(i, l) * dirac(j, k)))

    return C



class ElasticityOperator(NumpyMatrixBasedOperator):

    sparse = True

    def __init__(self, grid, boundary_info, material_elasticity_tensor,
                 dirichlet_clear_columns=True, dirichlet_clear_diag=False, name=None):
        assert grid.reference_element is triangle
        self.dim_source = self.dim_range = grid.size(2) * 2
        self.grid = grid
        self.boundary_info = boundary_info
        self.material_elasticity_tensor = material_elasticity_tensor.copy()
        self.dirichlet_clear_columns = dirichlet_clear_columns
        self.dirichlet_clear_diag = dirichlet_clear_diag
        self.name = name

    def _assemble(self, mu=None):
        assert self.check_parameter(mu)
        g = self.grid
        bi = self.boundary_info
        C = self.material_elasticity_tensor

        self.logger.info('Calulate gradients of shape functions transformed by reference map ...')
        SF_GRADS = np.einsum('eij,pkj->epki', g.jacobian_inverse_transposed(0), shape_function_gradients)

        self.logger.info('Calculate all local scalar products beween gradients ...')
        SF_INTS = np.einsum('ijkl,epij,eqkl,e->epq', C, SF_GRADS, SF_GRADS, g.volumes(0)).ravel()

        self.logger.info('Determine global dofs ...')
        L_TO_G = np.concatenate((g.subentities(0, 2) * 2, g.subentities(0, 2) * 2 + 1), 1)
        SF_I0 = np.repeat(L_TO_G, 6, axis=1).ravel()
        SF_I1 = np.tile(L_TO_G, [1, 6]).ravel()

        self.logger.info('Boundary treatment ...')
        if bi.has_dirichlet:
            SF_INTS = np.where(bi.dirichlet_mask(2)[SF_I0 // 2], 0, SF_INTS)
            if self.dirichlet_clear_columns:
                SF_INTS = np.where(bi.dirichlet_mask(2)[SF_I1 // 2], 0, SF_INTS)

            if not self.dirichlet_clear_diag:
                SF_INTS = np.hstack((SF_INTS, np.ones(bi.dirichlet_boundaries(2).size * 2)))
                SF_I0 = np.hstack((SF_I0, bi.dirichlet_boundaries(2) * 2, bi.dirichlet_boundaries(2) * 2 + 1))
                SF_I1 = np.hstack((SF_I1, bi.dirichlet_boundaries(2) * 2, bi.dirichlet_boundaries(2) * 2 + 1))

        self.logger.info('Assemble system matrix ...')
        A = coo_matrix((SF_INTS, (SF_I0, SF_I1)), shape=(g.size(g.dim) * 2, g.size(g.dim) * 2))
        A = csc_matrix(A).copy()

        # The call to copy() is necessary to resize the data arrays of the sparse matrix:
        # During the conversion to crs_matrix, entries corresponding with the same
        # coordinates are summed up, resulting in shorter data arrays. The shortening
        # is implemented by calling self.prune() which creates the view self.data[:self.nnz].
        # Thus, the original data array is not deleted and all memory stays allocated.

        # from pymor.tools.memory import print_memory_usage
        # print_memory_usage('matrix: {0:5.1f}'.format((A.data.nbytes + A.indptr.nbytes + A.indices.nbytes)/1024**2))

        return NumpyMatrixOperator(A)


class H1ProductOperator(ElasticityOperator):

    def __init__(self, grid, boundary_info,
                 dirichlet_clear_columns=True, dirichlet_clear_diag=False, name=None):

        def dirac(i, j):
            return 1 if i == j else 0

        C = np.zeros((dim,) * 4, dtype=np.float)
        for i, j, k, l in product(xrange(dim), repeat=4):
            C[i, j, k, l] = dirac(i, k) * dirac(j, l)

        super(H1ProductOperator, self).__init__(grid, boundary_info, C,
                                                dirichlet_clear_columns, dirichlet_clear_diag, name)
