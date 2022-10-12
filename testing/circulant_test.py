from time import perf_counter
from colorama import Fore

from src.forward_operator_class import *


INPUT_SIZE = (4, 4, 3)
SPECTRAL_STENCIL = np.linspace(4400, 6500, INPUT_SIZE[2])


def run_test(matrix):
    matrix = matrix.T @ matrix

    def column_to_array(j):
        return matrix[:, [j]].toarray().flatten()


    etalon = column_to_array(0)
    n = len(etalon)

    for j in range(matrix.shape[1]):
        tmp = column_to_array(j)

        if not np.array_equal(etalon[:n - j], tmp[j:]) and not np.array_equal(etalon[n - j:], tmp[:j]):
            return False

    return True


def cfa_test(cfa):
    start = perf_counter()

    if run_test(cfa_operator(cfa, INPUT_SIZE, SPECTRAL_STENCIL).get_matrix_operator()):
        duration = perf_counter() - start
        print(Fore.GREEN + f'CFA operator circulant test passed in {duration:.2f} seconds for the CFA {cfa}.')

    else:
        print(Fore.RED + f'CFA operator circulant test failed for the CFA {cfa}.')


def binning_test(cfa):
    start = perf_counter()

    if run_test(binning_operator(cfa, INPUT_SIZE[:2]).get_matrix_operator()):
        duration = perf_counter() - start
        print(Fore.GREEN + f'Binning operator circulant test passed in {duration:.2f} seconds for the CFA {cfa}.')

    else:
        print(Fore.RED + f'Binning operator circulant test failed for the CFA {cfa}.')


def forward_test(cfa, binning):
    start = perf_counter()

    if binning:
        postfix = ' with binning.'

    else:
        postfix = ' without binning.'

    if run_test(Forward_operator(cfa, binning, 0, INPUT_SIZE, SPECTRAL_STENCIL).get_matrix_operator()):
        duration = perf_counter() - start
        print(Fore.GREEN + f'Forward operator circulant test passed in {duration:.2f} seconds for the CFA {cfa}' + postfix)

    else:
        print(Fore.RED + f'Forward operator circulant test failed for the CFA {cfa}' + postfix)


def run_circulant_tests():
    print(Fore.YELLOW + '####################### Beginning of the circulant tests #######################')

    cfa_test('bayer')
    forward_test('bayer', False)

    cfa_test('quad_bayer')
    forward_test('quad_bayer', False)

    binning_test('quad_bayer')
    forward_test('quad_bayer', True)

    cfa_test('sparse_3')
    forward_test('sparse_3', False)

    print(Fore.YELLOW + '########################## End of the circulant tests ##########################')