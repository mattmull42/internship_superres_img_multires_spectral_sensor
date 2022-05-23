from src.operators.TV_ajdoint_class import *


class TV_operator(odl.Operator):
    def __init__(self, input_size):
        self.input_size = input_size

        odl.Operator.__init__(self, odl.uniform_discr([0, 0, 0], self.input_size, self.input_size), odl.ProductSpace(odl.rn(input_size[0] * input_size[1]), 2 * input_size[2]),)

        self.grad = odl.Gradient(odl.uniform_discr([0, 0], input_size[:2], input_size[:2]), pad_mode='order0')

        self.tmp = np.zeros((2, self.input_size[0], self.input_size[1], self.input_size[2]))

        self.adjoint_op = TV_adjoint(self.input_size)


    def _call(self, Ux):
        for k in range(self.input_size[2]):
            self.tmp[:, :, :, k] = self.grad(Ux[:, :, k]).asarray()

        self.output = reduce_dimensions(np.transpose(reduce_dimensions(np.transpose(self.tmp, (1, 2, 3, 0))), (1, 2, 0)))

        return self.output


    @property
    def adjoint(self):
        return self.adjoint_op