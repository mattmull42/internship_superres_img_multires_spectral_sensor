from src.operators.cfa_operator_class import *
from src.operators.binning_operator_class import *


class Forward_operator(odl.Operator):
    def __init__(self, cfa, input_size, spectral_stencil, binning):
        check_init_parameters(cfa=cfa)

        self.cfa = cfa
        self.binning = binning
        self.input_size = input_size

        self.cfa_operator = cfa_operator(self.cfa, self.input_size, spectral_stencil)

        if self.binning:
            self.binning_operator = binning_operator(self.cfa, self.input_size[:2])
            self.binning_factor = self.binning_operator.l
            self.output_size = self.binning_operator.P_i, self.binning_operator.P_j

        else:
            self.output_size = self.input_size[:2]

        if self.binning:
            if self.cfa == 'quad_bayer':
                self.adjoint_op = odl.OperatorComp(self.cfa_operator.adjoint, self.binning_operator.adjoint)

        else:
            self.adjoint_op = self.cfa_operator.adjoint

        odl.Operator.__init__(self, odl.uniform_discr([0, 0, 0], self.input_size, self.input_size), odl.uniform_discr([0, 0], self.output_size, self.output_size))


    def _call(self, Ux):
        if self.binning:
            if self.cfa == 'quad_bayer':
                self.output = self.binning_operator(self.cfa_operator(Ux))

        else:
            self.output = self.cfa_operator(Ux)

        return self.output


    @property
    def adjoint(self):
        return self.adjoint_op


    def get_parameters(self):
        if self.binning:
            return self.cfa_operator.cfa_mask, self.input_size[:2], self.binning_factor

        else:
            return [self.cfa_operator.cfa_mask]


    def save_output(self, input_name):
        create_output_dirs()

        self.cfa_operator.save_output(input_name)
        
        if self.binning:
            if self.cfa == 'quad_bayer':
                file_name = path.basename(path.splitext(input_name)[0])+ '_binned_' + self.cfa + '.png'
                self.binning_operator.save_output(input_name)

        else:
            file_name = path.basename(path.splitext(input_name)[0]) + '_' + self.cfa + '.png'

        plt.imsave(path.join('output', 'forward_model_outputs', file_name), self.output, cmap='gray')