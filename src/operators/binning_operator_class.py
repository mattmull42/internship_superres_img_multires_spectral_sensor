import odl
import matplotlib.pyplot as plt

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
        self.output = np.zeros(self.range.shape)

        for i in range(self.P_i):
            for j in range(self.P_j):
                l_i = ((i + self.l) >= self.input_size[0]) * (self.input_size[0] - i - self.l) + self.l
                l_j = ((j + self.l) >= self.input_size[1]) * (self.input_size[1] - j - self.l) + self.l
                
                self.output[i, j] = np.mean(X[i * self.l:i * self.l + l_i, j * self.l:j * self.l + l_j])

        return self.output


    @property
    def adjoint(self):
        return binning_adjoint(self.input_size, self.l)


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_binned.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_binning', file_name), self.output, cmap='gray')