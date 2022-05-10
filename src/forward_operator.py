from src.operators.cfa_operator_class import *
from src.operators.binning_operator import *
from src.operators.sub_sampling_operator import *


class Forward_operator(odl.Operator):
    def __init__(self, cfa, N_i, N_j, N_k, spectral_stencil, binning):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.binning = binning
        self.N_i = N_i
        self.N_j = N_j
        self.N_k = N_k

        self.cfa_operator = cfa_operator(self.cfa, self.N_i, self.N_j, self.N_k, spectral_stencil)

        if self.binning:
            self.binning_operator = binning_operator(self.cfa, self.N_i, self.N_j)
            self.sub_sampling_operator = sub_sampling_operator(self.cfa, self.N_i, self.N_j)
            self.binning_factor = self.sub_sampling_operator.l_GCD
            output_shape = self.sub_sampling_operator.P_i, self.sub_sampling_operator.P_j

        else:
            output_shape = (self.N_i, self.N_j)

        odl.Operator.__init__(self, odl.rn((self.N_i, self.N_j, self.N_k)), odl.rn(output_shape), linear=True)


    def _call(self, Ux):
        if self.binning:
            if self.cfa == 'quad_bayer':
                self.output = self.sub_sampling_operator(self.binning_operator(self.cfa_operator(Ux)))

        else:
            self.output = self.cfa_operator(Ux)

        return self.output


    def get_parameters(self):
        return self.N_i, self.N_j, self.binning_factor


    def save_output(self, input_name):
        create_output_dirs()

        if self.binning:
            if self.cfa == 'quad_bayer':
                file_name = path.basename(path.splitext(input_name)[0])+ '_binned_' + self.cfa + '.png'
                self.cfa_operator.save_output(input_name)
                self.binning_operator.save_output(input_name)
                self.sub_sampling_operator.save_output(input_name)

        else:
            file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '.png'

        plt.imsave(path.join('output', 'forward_model_outputs', file_name), self.output, cmap='gray')