import matplotlib.pyplot as plt
import torch
from scipy.sparse import coo_array

from src.utilities.tool_box import *


class binning_operator():
    def __init__(self, cfa, input_size):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.input_size = input_size

        if self.cfa == 'quad_bayer':
            self.l = 2

        self.P_i = int(np.ceil(self.input_size[0] / self.l))
        self.P_j = int(np.ceil(self.input_size[1] / self.l))

        self.adjoint_op = binning_adjoint(self.input_size, self.l)


    def __call__(self, X):
        arr = torch.tensor(np.expand_dims(X, axis=(0,1)))
        ker = torch.tensor(np.expand_dims(np.ones((2, 2)) / 4, axis=(0,1)))
        self.output = torch.nn.functional.conv2d(arr, ker, stride=2, padding=((self.input_size[0] % 2) * 2, (self.input_size[1] % 2) * 2)).numpy().squeeze()

        if self.input_size[0] % 2:
            self.output = self.output[1:]

        if self.input_size[1] % 2:
            self.output = self.output[:, 1:]

        return self.output


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

            self.matrix_operator = coo_array((binning_data, (binning_i, binning_j)), shape=(P_ij, N_ij))

        return self.matrix_operator


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_binned.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_binning', file_name), self.output, cmap='gray')


class binning_adjoint():
    def __init__(self, output_size, l_GCD):
        self.output_size = output_size
        self.l = l_GCD
        self.P_i = int(np.ceil(self.output_size[0] / self.l))
        self.P_j = int(np.ceil(self.output_size[1] / self.l))


    def __call__(self, X):
        self.output = np.repeat(np.repeat(X, 2, axis=0), 2, axis=1) / self.l**2

        if self.output_size[0] % 2:
            self.output = self.output[:-1]

        if self.output_size[1] % 2:
            self.output = self.output[:, :-1]

        return self.output