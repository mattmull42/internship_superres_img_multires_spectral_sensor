import odl

from src.utilities.tool_box import *


class TV_adjoint(odl.Operator):
    def __init__(self, output_size):
        self.output_size = output_size

        odl.Operator.__init__(self, odl.ProductSpace(odl.rn(output_size[0] * output_size[1]), 2 * output_size[2]), odl.uniform_discr([0, 0, 0], self.output_size, self.output_size))

        self.grad = odl.Gradient(odl.uniform_discr([0, 0], output_size[:2], output_size[:2]), pad_mode='order0')

        self.output = np.zeros(self.range.shape)


    def _call(self, UG):
        tmp = np.transpose(increase_dimensions(np.transpose(UG.asarray()), np.append(self.output_size, 2)), (3, 0, 1, 2))

        for k in range(self.output_size[2]):
            self.output[:, :, k] = self.grad.adjoint(tmp[:, :, :, k])

        return self.output