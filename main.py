import numpy as np
import matplotlib.pyplot as plt

from src.forward_operator.forward_operator import forward_operator
from src.forward_operator.operators import *

from src.input_initialization import initialize_input, initialize_inverse_input

from os import listdir


CFA = 'sparse_3'
BINNING = CFA == 'quad_bayer'
NOISE_LEVEL = 0

# INPUT_DIR = 'input/'
# x, spectral_stencil = initialize_input(INPUT_DIR + '01690.png')


INPUT_DIR = 'input/balloons_ms/'
spectral_stencil = np.array([i for i in range(400, 701, 10)])
x = np.empty((512, 512, 31))

for file in listdir(INPUT_DIR):
    if file.endswith('.png'):
        x[:, :, int(file[-6:-4]) - 1] = initialize_inverse_input(INPUT_DIR + file)

x /= np.max(x)


cfa_op = cfa_operator(CFA, x.shape, spectral_stencil, 'WV34bands_Spectral_Responses.npz')
forward_op = forward_operator([cfa_op])

y = forward_op.direct(x)


fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
rgb_window = [np.abs(spectral_stencil - 650).argmin(), np.abs(spectral_stencil - 525).argmin(), np.abs(spectral_stencil - 480).argmin()]
axs[0].imshow(x[:, :, rgb_window])
axs[1].imshow(y, cmap='gray')
plt.show()