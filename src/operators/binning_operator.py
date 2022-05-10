import odl
import matplotlib.pyplot as plt

from src.utilities.tool_box import *


class binning_operator(odl.Operator):
    def __init__(self, cfa, N_i, N_j):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.N_i = N_i
        self.N_j = N_j

        odl.Operator.__init__(self, odl.rn((self.N_i, self.N_j)), odl.rn((self.N_i, self.N_j)), linear=True)

        if self.cfa == 'quad_bayer':
            self.get_quad_binning_matrix()


    def _call(self, X):
        X = X.asarray()
        self.output = np.zeros((self.N_i, self.N_j))

        for i in range(self.N_i):
            for j in range(self.N_j):
                l = int(self.binning_matrix[i, j])
                
                if l != 0:
                    l_i = ((i + l) >= self.N_i) * (self.N_i - i - l) + l
                    l_j = ((j + l) >= self.N_j) * (self.N_j - j - l) + l
                    
                    self.output[i:i + l_i, j:j + l_j] = np.mean(X[i:i + l_i, j:j + l_j])

        return self.output


    def get_quad_binning_matrix(self):
        self.binning_matrix = np.zeros((self.N_i, self.N_j))

        for i in range(self.N_i):
            for j in range(self.N_j):
                if i % 2 == 0 and j % 2 == 0:
                    self.binning_matrix[i, j] = 2


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_binned.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_binning', file_name), self.output, cmap='gray')