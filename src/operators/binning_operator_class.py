import matplotlib.pyplot as plt
import torch
from torch.nn.functional import conv2d
import scipy.sparse as sp

from src.operators.binning_adjoint_class import *


class binning_operator(odl.Operator):
    def __init__(self, cfa, input_size):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.input_size = input_size

        if self.cfa == 'quad_bayer':
            self.l = 2

        self.P_i = int(np.ceil(self.input_size[0] / self.l))
        self.P_j = int(np.ceil(self.input_size[1] / self.l))

        odl.Operator.__init__(self, odl.uniform_discr([0, 0], self.input_size, self.input_size), odl.uniform_discr([0, 0], (self.P_i, self.P_j), (self.P_i, self.P_j)))

        self.adjoint_op = binning_adjoint(self.input_size, self.l)


    def _call(self, X):
        X = X.asarray()
        ker = np.ones((2, 2)) / 4

        arr = torch.tensor(np.expand_dims(X, axis=(0,1)))
        arr2 = torch.tensor(np.expand_dims(ker, axis=(0,1)))
        self.output = conv2d(arr, arr2, stride=2, padding=((self.input_size[0] % 2) * 2, (self.input_size[1] % 2) * 2)).numpy().squeeze()

        if self.input_size[0] % 2:
            self.output = self.output[1:]

        if self.input_size[1] % 2:
            self.output = self.output[:, 1:]

        return self.output


    @property
    def adjoint(self):
        return self.adjoint_op


    def get_matrix_operator(self):
        if not hasattr(self, 'matrix_operator'):
            N_ij = self.input_size[0] * self.input_size[1]

            if self.cfa == 'quad_bayer':
                P_ij = self.P_i * self.P_j

                binning_i, binning_j = [], []

                for i in range(P_ij):
                    tmp_i, tmp_j = 2 * (i % self.P_i), 2 * (i // self.P_i)
                    binning_i.append(i)
                    binning_j.append(tmp_i + self.input_size[0] * tmp_j)

                    if tmp_i + 1 < self.input_size[0]:
                        binning_i.append(i)
                        binning_j.append(tmp_i + 1 + self.input_size[0] * tmp_j)

                        if tmp_j + 1 < self.input_size[1]:
                            binning_i.append(i)
                            binning_i.append(i)
                            binning_j.append(tmp_i + self.input_size[0] * (tmp_j + 1))
                            binning_j.append(tmp_i + 1 + self.input_size[0] * (tmp_j + 1))

                binning_data = np.ones_like(binning_i) / 4

            self.matrix_operator = sp.csc_array((binning_data, (binning_i, binning_j)), shape=(P_ij, N_ij))

        return self.matrix_operator


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_binned.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_binning', file_name), self.output, cmap='gray')