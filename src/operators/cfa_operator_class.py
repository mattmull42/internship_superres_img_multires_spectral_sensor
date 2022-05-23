import matplotlib.pyplot as plt

from src.operators.cfa_adjoint_class import *


class cfa_operator(odl.Operator):
    def __init__(self, cfa, input_size, spectral_stencil):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.input_size = input_size
        self.k_r, self.k_g, self.k_b = get_indices_rgb(spectral_stencil)

        odl.Operator.__init__(self, odl.uniform_discr([0, 0, 0], self.input_size, self.input_size), odl.rn(self.input_size[:-1]))

        if self.cfa == 'bayer':
            self.get_bayer_mask()

        elif self.cfa == 'quad_bayer':
            self.get_quad_mask()

        self.adjoint_op = cfa_adjoint(self.input_size, self.cfa_mask)


    def _call(self, Ux):
        Ux = Ux.asarray()

        self.output = increase_dimensions(np.sum(reduce_dimensions(Ux) * self.cfa_mask, axis=1), self.input_size[:2])

        return self.output


    @property
    def adjoint(self):
        return self.adjoint_op


    def get_bayer_mask(self):
        self.cfa_mask = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2]))

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                if i % 2 == 0 and j % 2 == 1:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 1

                elif i % 2 == 1 and j % 2 == 0:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 1

                else:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 1



    def get_quad_mask(self):
        self.cfa_mask = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2]))

        for i in range(self.input_size[0]):
            for j in range(self.input_size[1]):
                if i % 4 < 2 and j % 4 >= 2:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_r] = 1

                elif i % 4 >= 2 and j % 4 < 2:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_b] = 1

                else:
                    self.cfa_mask[i + self.input_size[0] * j, self.k_g] = 1


    def save_output(self, input_name):
        create_output_dirs()
        file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '.png'
        plt.imsave(path.join('output', 'forward_model_outputs', 'output_cfa', file_name), self.output, cmap='gray')