import matplotlib.pyplot as plt
from scipy.sparse import coo_array

from src.operators.cfa_adjoint_class import *


class cfa_operator(odl.Operator):
    def __init__(self, cfa, input_size, spectral_stencil):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.input_size = input_size
        self.k_r, self.k_g, self.k_b = get_indices_rgb(spectral_stencil)

        odl.Operator.__init__(self, odl.uniform_discr([0, 0, 0], self.input_size, self.input_size), odl.rn(self.input_size[:-1]))

        if self.cfa == 'bayer':
            self.get_bayer_mask()

        elif self.cfa == 'quad_bayer':
            self.get_quad_mask()

        elif self.cfa == 'sparse_3':
            self.get_sparse_3_mask()

        self.adjoint_op = cfa_adjoint(self.input_size, self.cfa_mask)


    def _call(self, Ux):
        Ux = Ux.asarray()

        self.output = increase_dimensions(np.sum(reduce_dimensions(Ux) * self.cfa_mask, axis=1), self.input_size[:2])

        return self.output


    @property
    def adjoint(self):
        return self.adjoint_op


    def get_bayer_mask(self):
        self.cfa_mask = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2]))

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                if i % 2 == 0 and j % 2 == 1:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 1

                elif i % 2 == 1 and j % 2 == 0:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 1

                else:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 1



    def get_quad_mask(self):
        self.cfa_mask = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2]))

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                if i % 4 < 2 and j % 4 >= 2:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 1

                elif i % 4 >= 2 and j % 4 < 2:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 1

                else:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 1


    def get_sparse_3_mask(self):
        self.cfa_mask = np.full((self.input_size[0] * self.input_size[1], self.input_size[2]), 1 / self.input_size[2])

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                if i % 8 == 0 and j % 8 == 0:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 1
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 0
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 0

                elif i % 8 == 4 and j % 8 == 4:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 0
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 1
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 0

                elif (i % 8 == 4 and j % 8 == 0) or (i % 8 == 0 and j % 8 == 4):
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 0
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 0
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 1


    def get_matrix_operator(self):
        if not hasattr(self, 'matrix_operator'):
            N_ij = self.input_size[0] * self.input_size[1]
            N_ijk = self.input_size[0] * self.input_size[1] * self.input_size[2]

            cfa_i = np.repeat(list(range(N_ij)), self.input_size[2])
            cfa_j = []

            for i in range(N_ij):
                cfa_j += [i + k * N_ij for k in range(self.input_size[2])]

            cfa_data = [self.cfa_mask[cfa_i[a], cfa_j[a] // N_ij] for a in range(len(cfa_i))]

            self.matrix_operator = coo_array((cfa_data, (cfa_i, cfa_j)), shape=(N_ij, N_ijk))

        return self.matrix_operator


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_cfa', file_name), self.output, cmap='gray')