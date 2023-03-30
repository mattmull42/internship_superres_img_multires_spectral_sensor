import numpy as np

from .spectral_responses.get_spectral_responses import get_filter_response


def get_rbgp_bands(file_name):
    if file_name == 'WV34bands_Spectral_Responses.npz':
        return 2, 1, 0, 4

    return 'red', 'green', 'blue', 'pan'


def get_bayer_mask(input_shape, spectral_stencil, responses_file):
    band_r, band_g, band_b, _ = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), green_filter)

    cfa_mask[::2, 1::2] = red_filter
    cfa_mask[1::2, ::2] = blue_filter

    return cfa_mask


def get_quad_mask(input_shape, spectral_stencil, responses_file):
    band_r, band_g, band_b, _ = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), green_filter)

    cfa_mask[::4, 2::4] = red_filter
    cfa_mask[::4, 3::4] = red_filter
    cfa_mask[1::4, 2::4] = red_filter
    cfa_mask[1::4, 3::4] = red_filter

    cfa_mask[2::4, ::4] = blue_filter
    cfa_mask[2::4, 1::4] = blue_filter
    cfa_mask[3::4, ::4] = blue_filter
    cfa_mask[3::4, 1::4] = blue_filter

    return cfa_mask


def get_sparse_3_mask(input_shape, spectral_stencil, responses_file):
    band_r, band_g, band_b, band_p = get_rbgp_bands(responses_file)

    red_filter = get_filter_response(spectral_stencil, responses_file, band_r)
    green_filter = get_filter_response(spectral_stencil, responses_file, band_g)
    blue_filter = get_filter_response(spectral_stencil, responses_file, band_b)
    pan_filter = get_filter_response(spectral_stencil, responses_file, band_p)

    cfa_mask = np.kron(np.ones((input_shape[0], input_shape[1], 1)), pan_filter)

    cfa_mask[::8, ::8] = red_filter

    cfa_mask[::8, 4::8] = green_filter
    cfa_mask[4::8, ::8] = green_filter

    cfa_mask[4::8, 4::8] = blue_filter

    return cfa_mask