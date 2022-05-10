import odl
import matplotlib.pyplot as plt

from src.utilities.tool_box import *


class sub_sampling_operator(odl.Operator):
    def __init__(self, cfa, N_i, N_j):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.N_i = N_i
        self.N_j = N_j

        if self.cfa == 'quad_bayer':
            self.l_GCD = 2

        self.P_i = int(np.ceil(self.N_i / self.l_GCD))
        self.P_j = int(np.ceil(self.N_j / self.l_GCD))

        odl.Operator.__init__(self, odl.rn((self.N_i, self.N_j)), odl.rn((self.P_i, self.P_j)), linear=True)


    def _call(self, X):
        X = X.asarray()
        self.output = np.zeros((self.P_i, self.P_j))

        for i in range(self.P_i):
            for j in range(self.P_j):
                self.output[i, j] = X[i * self.l_GCD, j * self.l_GCD]

        return self.output


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_subsampled.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_subsampling', file_name), self.output, cmap='gray')