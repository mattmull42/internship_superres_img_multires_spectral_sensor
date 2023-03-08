import numpy as np


def get_indices_rgb(spectral_stencil):
    stencil = np.array(spectral_stencil)

    return (np.abs(stencil - 650)).argmin(), (np.abs(stencil - 550)).argmin(), (np.abs(stencil - 445)).argmin()


def get_bayer_mask(input_size, spectral_stencil):
    k_r, k_g, k_b = get_indices_rgb(spectral_stencil)

    cfa_mask = np.zeros(input_size)

    cfa_mask[::2, 1::2, k_r] = 1
    cfa_mask[1::2, ::2, k_b] = 1
    cfa_mask[::2, ::2, k_g] = 1
    cfa_mask[1::2, 1::2, k_g] = 1

    return cfa_mask


def get_quad_mask(input_size, spectral_stencil):
    k_r, k_g, k_b = get_indices_rgb(spectral_stencil)

    cfa_mask = np.zeros(input_size)

    cfa_mask[::4, 2::4, k_r] = 1
    cfa_mask[::4, 3::4, k_r] = 1
    cfa_mask[1::4, 2::4, k_r] = 1
    cfa_mask[1::4, 3::4, k_r] = 1

    cfa_mask[::4, ::4, k_g] = 1
    cfa_mask[::4, 1::4, k_g] = 1
    cfa_mask[1::4, ::4, k_g] = 1
    cfa_mask[1::4, 1::4, k_g] = 1
    cfa_mask[2::4, 2::4, k_g] = 1
    cfa_mask[2::4, 3::4, k_g] = 1
    cfa_mask[3::4, 2::4, k_g] = 1
    cfa_mask[3::4, 3::4, k_g] = 1

    cfa_mask[2::4, ::4, k_b] = 1
    cfa_mask[2::4, 1::4, k_b] = 1
    cfa_mask[3::4, ::4, k_b] = 1
    cfa_mask[3::4, 1::4, k_b] = 1

    return cfa_mask


def get_sparse_3_mask(input_size, spectral_stencil):
    k_r, k_g, k_b = get_indices_rgb(spectral_stencil)

    cfa_mask = np.full(input_size, 1 / input_size[2])

    cfa_mask[::8, ::8, k_r] = 1
    cfa_mask[::8, ::8, :k_r] = 0
    cfa_mask[::8, ::8, k_r + 1:] = 0

    cfa_mask[::8, 4::8, k_g] = 1
    cfa_mask[::8, 4::8, :k_g] = 0
    cfa_mask[::8, 4::8, k_g + 1:] = 0
    cfa_mask[4::8, ::8, k_g] = 1
    cfa_mask[4::8, ::8, :k_g] = 0
    cfa_mask[4::8, ::8, k_g + 1:] = 0

    cfa_mask[4::8, 4::8, k_b] = 1
    cfa_mask[4::8, 4::8, :k_b] = 0
    cfa_mask[4::8, 4::8, k_b + 1:] = 0

    return cfa_mask