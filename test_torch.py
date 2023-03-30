import torch
import numpy as np
import matplotlib.pyplot as plt

from src.forward_operator.forward_operator import forward_operator
from src.forward_operator.operators import *

from src.input_initialization import initialize_input


INPUT_DIR = 'input/'

CFA = 'bayer'
BINNING = CFA == 'quad_bayer'

input_name = '01690'
gt, spectral_stencil = initialize_input(INPUT_DIR + input_name + '.png')


cfa_op = cfa_operator(CFA, gt.shape, spectral_stencil)
forward_op = forward_operator([cfa_op])
A = forward_op.matrix
A_t = torch.sparse_csr_tensor(A.indptr, A.indices, A.data)

y = (A @ gt.reshape(-1)).reshape(forward_op.output_shape)

gt = torch.tensor(gt.reshape(-1), requires_grad=True)
y_t = A_t @ gt

print(gt.grad)
y_t.sum().backward()
print(gt.grad.sum())

# fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# axs[0].imshow(gt)
# axs[1].imshow(y, cmap='gray')
# axs[2].imshow(y_t.reshape(forward_op.output_shape), cmap='gray')
# plt.show()