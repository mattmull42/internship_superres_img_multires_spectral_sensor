from .cfa_operator import *
from .binning_operator import *


class forward_operator():
    def __init__(self, cfa, binning, noise_level, input_size, spectral_stencil):
        self.cfa = cfa
        self.binning = binning
        self.input_size = input_size
        self.noise_level = noise_level

        self.cfa_operator = cfa_operator(self.cfa, self.input_size, spectral_stencil)

        if self.binning:
            self.binning_operator = binning_operator(self.cfa, self.input_size[:2])
            self.binning_factor = self.binning_operator.l
            self.output_size = self.binning_operator.P_i, self.binning_operator.P_j

        else:
            self.output_size = self.input_size[:2]


    def __call__(self, x):
        self.output = self.cfa_operator(x)
        
        if self.noise_level != 0:
            self.add_noise()

        if self.binning:
            if self.cfa == 'quad_bayer':
                self.output = self.binning_operator(self.output)

        return self.output


    def adjoint(self, y):
        if self.binning:
            self.adjoint_output = self.cfa_operator.adjoint(self.binning_operator.adjoint(y))

        else:
            self.adjoint_output = self.cfa_operator.adjoint(y)

        return self.adjoint_output


    def add_noise(self):
        self.output = np.clip(self.output + np.random.normal(0, self.noise_level / 100, self.output.shape), 0, 1)


    def get_matrix(self):
        if not hasattr(self, 'matrix'):
            if self.binning:
                self.matrix = self.binning_operator.get_matrix() @ self.cfa_operator.get_matrix()

            else:
                self.matrix = self.cfa_operator.get_matrix()

        return self.matrix