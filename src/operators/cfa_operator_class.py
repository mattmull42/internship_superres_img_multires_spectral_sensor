import odl
import matplotlib.pyplot as plt

from src.utilities.tool_box import *


class cfa_operator(odl.Operator):
    def __init__(self, cfa, N_i, N_j, N_k, spectral_stencil):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.N_i = N_i
        self.N_j = N_j
        self.N_k = N_k
        self.spectral_stencil = spectral_stencil
        self.k_r, self.k_g, self.k_b = get_indices_rgb(self.spectral_stencil)

        odl.Operator.__init__(self, odl.rn((self.N_i, self.N_j, self.N_k)), odl.rn((self.N_i, self.N_j)), linear=True)

        if self.cfa == 'bayer':
            self.get_bayer_mask()

        elif self.cfa == 'quad_bayer':
            self.get_quad_mask()


    def _call(self, Ux):
        Ux = Ux.asarray()

        self.output = increase_dimensions(sum(reduce_dimensions(Ux)[:, k] * self.cfa_mask[:, k] for k in range(self.N_k)), (self.N_i, self.N_j))

        return self.output


    def get_bayer_mask(self):
        self.cfa_mask = np.zeros((self.N_i * self.N_j, self.N_k))

        for i in range(self.N_i):
            for j in range(self.N_j):
                for k in range(self.N_k):
                    if k == self.k_r:
                        if i % 2 == 0 and j % 2 == 1:
                            self.cfa_mask[i + self.N_i * j, k] = 1

                    elif k == self.k_g:
                        if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                            self.cfa_mask[i + self.N_i * j, k] = 1

                    elif k == self.k_b:
                        if i % 2 == 1 and j % 2 == 0:
                            self.cfa_mask[i + self.N_i * j, k] = 1


    def get_quad_mask(self):
        self.cfa_mask = np.zeros((self.N_i * self.N_j, self.N_k))

        for i in range(self.N_i):
            for j in range(self.N_j):
                for k in range(self.N_k):
                    if k == self.k_r:
                        if i % 4 < 2 and j % 4 >= 2:
                            self.cfa_mask[i + self.N_i * j, k] = 1

                    elif k == self.k_g:
                        if (i % 4 < 2 and j % 4 < 2) or (i % 4 >= 2 and j % 4 >= 2):
                            self.cfa_mask[i + self.N_i * j, k] = 1

                    elif k == self.k_b:
                        if i % 4 >= 2 and j % 4 < 2:
                            self.cfa_mask[i + self.N_i * j, k] = 1


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '_CFA.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_cfa', file_name), self.output, cmap='gray')