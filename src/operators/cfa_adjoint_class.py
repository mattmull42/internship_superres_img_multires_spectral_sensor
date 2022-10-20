import odl

from src.utilities.tool_box import *


class cfa_adjoint(odl.Operator):
    def __init__(self, output_size, cfa_mask):
        self.output_size = output_size
        self.cfa_mask = cfa_mask

        odl.Operator.__init__(self, odl.uniform_discr([0, 0], self.output_size[:2], self.output_size[:2]), odl.uniform_discr([0, 0, 0], self.output_size, self.output_size))


    def _call(self, X):
        self.output = self.cfa_mask * X.asarray()[..., np.newaxis]

        return self.output