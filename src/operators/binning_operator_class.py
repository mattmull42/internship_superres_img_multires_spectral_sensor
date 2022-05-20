import odl
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import conv2d

from src.utilities.tool_box import *
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

        odl.Operator.__init__(self, odl.uniform_discr(min_pt=[0, 0], max_pt=self.input_size, shape=self.input_size), odl.uniform_discr(min_pt=[0, 0], max_pt=(self.P_i, self.P_j), shape=(self.P_i, self.P_j)), linear=True)


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
        return binning_adjoint(self.input_size, self.l)


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_binned.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_binning', file_name), self.output, cmap='gray')