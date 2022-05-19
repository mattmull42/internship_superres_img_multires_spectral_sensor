import odl

from src.utilities.tool_box import *


class binning_adjoint(odl.Operator):
    def __init__(self, output_size, l_GCD):
        self.output_size = output_size
        self.l = l_GCD
        self.P_i = int(np.ceil(self.output_size[0] / self.l))
        self.P_j = int(np.ceil(self.output_size[1] / self.l))

        odl.Operator.__init__(self, odl.rn((self.P_i, self.P_j)), odl.uniform_discr(min_pt=[0, 0], max_pt=self.output_size, shape=self.output_size), linear=True)


    def _call(self, X):
        X = X.asarray()
        self.output = np.zeros(self.range.shape)

        for i in range(self.P_i):
            for j in range(self.P_j):
                l_i = ((i + self.l) >= self.output_size[0]) * (self.output_size[0] - i - self.l) + self.l
                l_j = ((j + self.l) >= self.output_size[1]) * (self.output_size[1] - j - self.l) + self.l

                self.output[i * self.l:i * self.l + l_i, j * self.l:j * self.l + l_j] = X[i, j] / self.l**2

        return self.output