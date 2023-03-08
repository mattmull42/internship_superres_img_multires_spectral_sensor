import numpy as np
from scipy.sparse import csc_array

from .abstract_operator import abstract_operator
from .misc.cfa_masks import get_bayer_mask, get_quad_mask, get_sparse_3_mask


class cfa_operator(abstract_operator):
    def __init__(self, cfa, input_shape, spectral_stencil, name=None):
        self.cfa = cfa
        self.name = 'CFA' if name is None else name

        if self.cfa == 'bayer':
            self.cfa_mask = get_bayer_mask(input_shape, spectral_stencil)

        elif self.cfa == 'quad_bayer':
            self.cfa_mask = get_quad_mask(input_shape, spectral_stencil)

        elif self.cfa == 'sparse_3':
            self.cfa_mask = get_sparse_3_mask(input_shape, spectral_stencil)

        super().__init__(input_shape, input_shape[:-1], self.name)


    def direct(self, x):
        return np.sum(x * self.cfa_mask, axis=2)


    def adjoint(self, y):
        return self.cfa_mask * y[..., np.newaxis]


    @property
    def matrix(self):
        N_k = self.input_shape[2]
        N_ij = self.input_shape[0] * self.input_shape[1]
        N_ijk = self.input_shape[0] * self.input_shape[1] * N_k

        cfa_i = np.repeat(np.arange(N_ij), N_k)
        cfa_j = np.arange(N_ijk)

        cfa_data = self.cfa_mask[cfa_i // self.input_shape[1], cfa_i % self.input_shape[1], cfa_j % N_k]

        return csc_array((cfa_data, (cfa_i, cfa_j)), shape=(N_ij, N_ijk))