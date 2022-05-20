import odl

from src.utilities.tool_box import *


class cfa_adjoint(odl.Operator):
    def __init__(self, output_size, cfa_mask):
        self.output_size = output_size
        self.cfa_mask = cfa_mask

        odl.Operator.__init__(self, odl.uniform_discr(min_pt=[0, 0], max_pt=self.output_size[:2], shape=self.output_size[:2]), odl.uniform_discr(min_pt=[0, 0, 0], max_pt=self.output_size, shape=self.output_size), linear=True)


    def _call(self, X):
        X = X.asarray()

        self.output = increase_dimensions(np.transpose(np.transpose(self.cfa_mask) * reduce_dimensions(X)), self.output_size)

        return self.output