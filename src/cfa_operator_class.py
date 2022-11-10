import matplotlib.pyplot as plt
from scipy.sparse import coo_array

from src.utilities.tool_box import *
from src.utilities.cfa_masks import *


class cfa_operator():
    def __init__(self, cfa, input_size, spectral_stencil):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.input_size = input_size
        self.k_r, self.k_g, self.k_b = get_indices_rgb(spectral_stencil)

        if self.cfa == 'bayer':
            self.cfa_mask = get_bayer_mask(self.input_size, self.k_r, self.k_g, self.k_b)

        elif self.cfa == 'quad_bayer':
            self.cfa_mask = get_quad_mask(self.input_size, self.k_r, self.k_g, self.k_b)

        elif self.cfa == 'sparse_3':
            self.cfa_mask = get_sparse_3_mask(self.input_size, self.k_r, self.k_g, self.k_b)

        self.adjoint_op = cfa_adjoint(self.input_size, self.cfa_mask)


    def __call__(self, Ux):
        self.output = np.sum(Ux * self.cfa_mask, axis=2)

        return self.output


    def get_matrix_operator(self):
        if not hasattr(self, 'matrix_operator'):
            N_ij = self.input_size[0] * self.input_size[1]
            N_ijk = self.input_size[0] * self.input_size[1] * self.input_size[2]

            cfa_i = np.repeat(list(range(N_ij)), self.input_size[2])
            cfa_j = []

            for i in range(N_ij):
                cfa_j += [i + k * N_ij for k in range(self.input_size[2])]

            cfa_data = [self.cfa_mask[cfa_i[a] % self.input_size[0], cfa_i[a] // self.input_size[0], cfa_j[a] // N_ij] for a in range(len(cfa_i))]

            self.matrix_operator = coo_array((cfa_data, (cfa_i, cfa_j)), shape=(N_ij, N_ijk))

        return self.matrix_operator


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_cfa', file_name), self.output, cmap='gray')


class cfa_adjoint():
    def __init__(self, output_size, cfa_mask):
        self.output_size = output_size
        self.cfa_mask = cfa_mask


    def __call__(self, X):
        self.output = self.cfa_mask * X[..., np.newaxis]

        return self.output