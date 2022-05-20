import odl

from src.utilities.tool_box import *


class binning_adjoint(odl.Operator):
    def __init__(self, output_size, l_GCD):
        self.output_size = output_size
        self.l = l_GCD
        self.P_i = int(np.ceil(self.output_size[0] / self.l))
        self.P_j = int(np.ceil(self.output_size[1] / self.l))

        odl.Operator.__init__(self, odl.rn((self.P_i, self.P_j)), odl.uniform_discr([0, 0], self.output_size, self.output_size))


    def _call(self, X):
        X = X.asarray()

        self.output = np.repeat(np.repeat(X, 2, axis=0), 2, axis=1) / self.l**2

        if self.output_size[0] % 2:
            self.output = self.output[:-1]

        if self.output_size[1] % 2:
            self.output = self.output[:, :-1]

        return self.output