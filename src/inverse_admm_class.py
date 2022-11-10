from proximal import *

from src.forward_operator_class import *


class Inverse_problem_ADMM:
    def __init__(self, cfa, binning, noise_level, output_size, spectral_stencil, max_iter, eps):
        check_init_parameters(cfa=cfa, binning=binning)

        self.output_size = output_size
        self.max_iter = max_iter
        self.eps = eps

        self.forward_op = forward_operator(cfa, binning, noise_level, output_size, spectral_stencil)


    def __call__(self, y):
        x = Variable(self.output_size)

        problem = Problem(sum_squares(forward_operator_proximal(self.forward_op, x) - y) + self.eps * group_norm1(grad(x, dims=2), group_dims=[2]))
        problem.solve(solver='admm', max_iters=self.max_iter)

        np.clip(x.value, 0, 1, x.value)

        self.output = x.value

        return self.output


    def save_output(self, input_name):
        create_output_dirs()
        output_dir = path.join('output', 'inverse_problem_ADMM_outputs')
        input_name_without_extension = path.basename(path.splitext(input_name)[0])

        plt.imsave(path.join(output_dir, input_name_without_extension + '_reconstructed_ADMM.png'), self.output)


class forward_operator_proximal(lin_ops.lin_op.LinOp):
    def __init__(self, forward_op, Ux):
        self.forward_op = forward_op

        super(forward_operator_proximal, self).__init__([Ux], self.forward_op.output_size, None)


    def forward(self, x, y):
        np.copyto(y[0], self.forward_op(x[0]))


    def adjoint(self, u, v):
        np.copyto(v[0], self.forward_op.adjoint_op(u[0]))