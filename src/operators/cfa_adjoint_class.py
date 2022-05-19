import odl

from src.utilities.tool_box import *


class cfa_adjoint(odl.Operator):
    def __init__(self, cfa, output_size, spectral_stencil):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.output_size = output_size
        self.spectral_stencil = spectral_stencil
        self.k_r, self.k_g, self.k_b = get_indices_rgb(self.spectral_stencil)

        odl.Operator.__init__(self, odl.uniform_discr(min_pt=[0, 0], max_pt=self.output_size[:2], shape=self.output_size[:2]), odl.uniform_discr(min_pt=[0, 0, 0], max_pt=self.output_size, shape=self.output_size), linear=True)


    def _call(self, X):
        X = X.asarray()
        self.output = np.zeros(self.range.shape)

        if self.cfa == 'bayer':
            self.apply_bayer_sparse(X)

        elif self.cfa == 'quad_bayer':
            self.apply_quad_sparse(X)

        return self.output


    def apply_bayer_sparse(self, X):
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                if i % 2 == 0 and j % 2 == 1:
                    self.output[i, j, self.k_r] = X[i, j]

                elif (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    self.output[i, j, self.k_g] = X[i, j]

                elif i % 2 == 1 and j % 2 == 0:
                    self.output[i, j, self.k_b] = X[i, j]


    def apply_quad_sparse(self, X):
        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                if i % 4 < 2 and j % 4 >= 2:
                    self.output[i, j, self.k_r] = X[i, j]

                elif (i % 4 < 2 and j % 4 < 2) or (i % 4 >= 2 and j % 4 >= 2):
                    self.output[i, j, self.k_g] = X[i, j]

                elif i % 4 >= 2 and j % 4 < 2:
                    self.output[i, j, self.k_b] = X[i, j]