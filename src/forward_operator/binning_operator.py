import torch
from scipy.sparse import csc_array
import numpy as np


class binning_operator():
    def __init__(self, cfa, input_size):
        self.cfa = cfa
        self.input_size = input_size

        if self.cfa == 'quad_bayer':
            self.l = 2

        self.P_i = int(np.ceil(self.input_size[0] / self.l))
        self.P_j = int(np.ceil(self.input_size[1] / self.l))


    def __call__(self, x):
        arr = torch.tensor(np.expand_dims(x, axis=(0,1)))
        ker = torch.tensor(np.expand_dims(np.ones((2, 2)) / 4, axis=(0,1)))
        self.output = torch.nn.functional.conv2d(arr, ker, stride=2, padding=((self.input_size[0] % 2) * 2, (self.input_size[1] % 2) * 2)).numpy().squeeze()

        if self.input_size[0] % 2:
            self.output = self.output[1:]

        if self.input_size[1] % 2:
            self.output = self.output[:, 1:]

        return self.output


    def adjoint(self, y):
        self.adjoint_output = np.repeat(np.repeat(y, 2, axis=0), 2, axis=1) / self.l**2

        if self.input_size[0] % 2:
            self.adjoint_output = self.adjoint_output[:-1]

        if self.input_size[1] % 2:
            self.adjoint_output = self.adjoint_output[:, :-1]

        return self.adjoint_output


    def get_matrix(self):
        if not hasattr(self, 'matrix'):
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

            self.matrix = csc_array((binning_data, (binning_i, binning_j)), shape=(P_ij, N_ij))

        return self.matrix