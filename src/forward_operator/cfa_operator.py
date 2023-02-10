from scipy.sparse import csc_array

from .cfa_masks import *


def get_indices_rgb(spectral_stencil):
    return (np.abs(spectral_stencil - 6500)).argmin(), (np.abs(spectral_stencil - 5500)).argmin(), (np.abs(spectral_stencil - 4450)).argmin()


class cfa_operator():
    def __init__(self, cfa, input_size, spectral_stencil):
        self.cfa = cfa
        self.input_size = input_size
        self.k_r, self.k_g, self.k_b = get_indices_rgb(spectral_stencil)

        if self.cfa == 'bayer':
            self.cfa_mask = get_bayer_mask(self.input_size, self.k_r, self.k_g, self.k_b)

        elif self.cfa == 'quad_bayer':
            self.cfa_mask = get_quad_mask(self.input_size, self.k_r, self.k_g, self.k_b)

        elif self.cfa == 'sparse_3':
            self.cfa_mask = get_sparse_3_mask(self.input_size, self.k_r, self.k_g, self.k_b)


    def __call__(self, x):
        self.output = np.sum(x * self.cfa_mask, axis=2)

        return self.output


    def adjoint(self, y):
        self.adjoint_output = self.cfa_mask * y[..., np.newaxis]

        return self.adjoint_output


    def get_matrix(self):
        if not hasattr(self, 'matrix'):
            N_k = self.input_size[2]
            N_ij = self.input_size[0] * self.input_size[1]
            N_ijk = self.input_size[0] * self.input_size[1] * N_k

            cfa_i = np.repeat(np.arange(N_ij), N_k)
            cfa_j = np.repeat(np.arange(N_ij), N_k) + N_ij * np.tile(np.arange(N_k), N_ij)

            cfa_data = self.cfa_mask[cfa_i % self.input_size[0], cfa_i // self.input_size[0], cfa_j // N_ij]

            self.matrix = csc_array((cfa_data, (cfa_i, cfa_j)), shape=(N_ij, N_ijk))

        return self.matrix