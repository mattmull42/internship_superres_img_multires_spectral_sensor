from src.forward_operator_class import *
from src.operators.TV_operator_class import *


class Inverse_problem_ADMM:
    def __init__(self, cfa, binning, noise_level, output_size, spectral_stencil, niter, sigma, eps, box_constraint):
        check_init_parameters(cfa=cfa, binning=binning)

        self.cfa = cfa
        self.binning = binning
        self.noise_level = noise_level
        self.output_size = output_size
        self.spectral_stencil = spectral_stencil
        self.box_constraint = box_constraint

        self.A = Forward_operator(self.cfa, self.binning, self.noise_level, self.output_size, self.spectral_stencil)
        self.TV = TV_operator(self.output_size)
        self.L = odl.BroadcastOperator(self.A, self.TV)

        self.sigma = sigma
        self.eps = eps

        self.f = odl.solvers.IndicatorBox(self.L.domain, 0, 1)

        if self.box_constraint:
            self.op_norm = 1.1 * odl.power_method_opnorm(self.L, maxiter=50)
            self.tau = self.sigma / self.op_norm ** 2

        else:
            self.tau = 1

        self.niter = niter


    def __call__(self, y, x=None):
        if x is not None:
            self.output = x

        else:
            if self.binning and self.cfa == 'quad_bayer':
                self.output = self.A.adjoint(y) * self.A.binning_factor**2

            else:
                self.output = self.A.adjoint(y)

        self.data_fit = odl.solvers.L2NormSquared(self.A.range).translated(y)
        self.reg_func = self.eps * odl.solvers.GroupL1Norm(self.TV.range, exponent=2)
        self.g = odl.solvers.SeparableSum(self.data_fit, self.reg_func)

        odl.solvers.admm_linearized(self.output, self.f, self.g, self.L, self.tau, self.sigma, self.niter)

        self.output = self.output.asarray()

        if not self.box_constraint:
            np.clip(self.output, 0, 1, self.output)

        return self.output


    def save_output(self, input_name):
        create_output_dirs()
        output_dir = path.join('output', 'inverse_problem_ADMM_outputs')
        input_name_without_extension = path.basename(path.splitext(input_name)[0])

        plt.imsave(path.join(output_dir, input_name_without_extension + '_reconstructed_ADMM.png'), self.output)