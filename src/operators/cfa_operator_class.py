import matplotlib.pyplot as plt

from src.operators.cfa_adjoint_class import *


class cfa_operator(odl.Operator):
    def __init__(self, cfa, input_size, spectral_stencil):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.input_size = input_size
        self.spectral_stencil = spectral_stencil
        self.k_r, self.k_g, self.k_b = get_indices_rgb(self.spectral_stencil)

        odl.Operator.__init__(self, odl.uniform_discr(min_pt=[0, 0, 0], max_pt=self.input_size, shape=self.input_size), odl.rn(self.input_size[:-1]), linear=True)

        if self.cfa == 'bayer':
            self.get_bayer_mask()

        elif self.cfa == 'quad_bayer':
            self.get_quad_mask()


    def _call(self, Ux):
        Ux = Ux.asarray()

        self.output = increase_dimensions(sum(reduce_dimensions(Ux)[:, k] * self.cfa_mask[:, k] for k in range(self.input_size[2])), self.input_size[:2])

        return self.output


    @property
    def adjoint(self):
        return cfa_adjoint(self.cfa, self.input_size, self.spectral_stencil)


    def get_bayer_mask(self):
        self.cfa_mask = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2]))

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                for k in range(self.input_size[2]):
                    if k == self.k_r:
                        if i % 2 == 0 and j % 2 == 1:
                            self.cfa_mask[i + self.input_size[0] * j, k] = 1

                    elif k == self.k_g:
                        if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                            self.cfa_mask[i + self.input_size[0] * j, k] = 1

                    elif k == self.k_b:
                        if i % 2 == 1 and j % 2 == 0:
                            self.cfa_mask[i + self.input_size[0] * j, k] = 1


    def get_quad_mask(self):
        self.cfa_mask = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2]))

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                for k in range(self.input_size[2]):
                    if k == self.k_r:
                        if i % 4 < 2 and j % 4 >= 2:
                            self.cfa_mask[i + self.input_size[0] * j, k] = 1

                    elif k == self.k_g:
                        if (i % 4 < 2 and j % 4 < 2) or (i % 4 >= 2 and j % 4 >= 2):
                            self.cfa_mask[i + self.input_size[0] * j, k] = 1

                    elif k == self.k_b:
                        if i % 4 >= 2 and j % 4 < 2:
                            self.cfa_mask[i + self.input_size[0] * j, k] = 1


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_cfa', file_name), self.output, cmap='gray')