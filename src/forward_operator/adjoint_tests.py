from time import perf_counter
from colorama import Fore

from .forward_operator import *


INPUT_SIZE = (513, 1043, 11)
SPECTRAL_STENCIL = np.linspace(4400, 6500, INPUT_SIZE[2])


def run_test(operator, x, y):
    return np.abs(np.sum(y * operator(x)) - np.sum(operator.adjoint(y) * x)) < 1e-9


def cfa_test(cfa):
    start = perf_counter()

    operator = cfa_operator(cfa, INPUT_SIZE, SPECTRAL_STENCIL)

    x = np.random.rand(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
    y = np.random.rand(INPUT_SIZE[0], INPUT_SIZE[1])

    if run_test(operator, x, y):
        duration = perf_counter() - start
        print(Fore.GREEN + f'CFA operator adjoint test passed in {duration:.2f} seconds for the CFA {cfa}.' + Fore.WHITE)

    else:
        print(Fore.RED + f'CFA operator adjoint test failed for the CFA {cfa}.' + Fore.WHITE)


def binning_test(cfa):
    start = perf_counter()

    operator = binning_operator(cfa, INPUT_SIZE[:2])

    x = np.random.rand(INPUT_SIZE[0], INPUT_SIZE[1])
    y = np.random.rand(operator.P_i, operator.P_j)

    if run_test(operator, x, y):
        duration = perf_counter() - start
        print(Fore.GREEN + f'Binning operator adjoint test passed in {duration:.2f} seconds for the CFA {cfa}.' + Fore.WHITE)

    else:
        print(Fore.RED + f'Binning operator adjoint test failed for the CFA {cfa}.' + Fore.WHITE)


def forward_test(cfa, binning):
    start = perf_counter()

    operator = forward_operator(cfa, binning, 0, INPUT_SIZE, SPECTRAL_STENCIL)

    x = np.random.rand(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
    y = np.random.rand(operator.output_size[0],operator.output_size[1])

    if binning:
        postfix = ' with binning.'

    else:
        postfix = ' without binning.'

    if run_test(operator, x, y):
        duration = perf_counter() - start
        print(Fore.GREEN + f'Forward operator adjoint test passed in {duration:.2f} seconds for the CFA {cfa}' + postfix + Fore.WHITE)

    else:
        print(Fore.RED + f'Forward operator adjoint test failed for the CFA {cfa}' + postfix + Fore.WHITE)


def run_adjoint_tests():
    print(Fore.YELLOW + '######################## Beginning of the adjoint tests ########################' + Fore.WHITE)

    cfa_test('bayer')
    forward_test('bayer', False)

    cfa_test('quad_bayer')
    forward_test('quad_bayer', False)

    binning_test('quad_bayer')
    forward_test('quad_bayer', True)

    cfa_test('sparse_3')
    forward_test('sparse_3', False)

    print(Fore.YELLOW + '########################### End of the adjoint tests ###########################' + Fore.WHITE)