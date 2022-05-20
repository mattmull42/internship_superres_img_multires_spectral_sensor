from src.operators.TV_ajoint_class import *


class TV_operator(odl.Operator):
    def __init__(self, input_size):
        self.input_size = input_size

        odl.Operator.__init__(self, odl.uniform_discr(min_pt=[0, 0, 0], max_pt=self.input_size, shape=self.input_size), odl.ProductSpace(odl.rn(input_size[0] * input_size[1]), 2 * input_size[2]), linear=True)

        self.grad = odl.Gradient(odl.uniform_discr(min_pt=[0, 0], max_pt=input_size[:2], shape=input_size[:2]), pad_mode='order0')


    def _call(self, Ux):
        tmp = np.zeros((self.input_size[0] * self.input_size[1], self.input_size[2], 2))

        for k in range(self.input_size[2]):
            grad = self.grad(Ux[:, :, k]).asarray()
            tmp[:, k, :] = reduce_dimensions(np.transpose(grad, (1, 2, 0)))

        self.output = reduce_dimensions(np.transpose(tmp, (1, 2, 0)))

        return self.output


    @property
    def adjoint(self):
        return TV_adjoint(self.input_size)