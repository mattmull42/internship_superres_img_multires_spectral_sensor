import matplotlib.pyplot as plt

from src.utilities.tool_box import *


class Forward_model:
    def __init__(self, cfa, binning):
        check_init_parameters(cfa, binning)

        self.cfa = cfa
        self.binning = binning


    def set_input(self, image, spectral_stencil, input_name):
        self.input = image
        self.spectral_stencil = spectral_stencil
        self.input_name = input_name
        self.k_r, self.k_g, self.k_b = get_indices_rgb(self.spectral_stencil)

        if len(self.input.shape) != 3:
            raise Exception('Input must be a 3 dimensional tensor.')

        self.N_i = self.input.shape[0]
        self.N_j = self.input.shape[1]
        self.N_k = self.input.shape[2]


    def run(self):
        if self.cfa == 'bayer':
            self.cfa_mask = self.get_bayer_mask()

        elif self.cfa == 'quad_bayer':
            self.cfa_mask = self.get_quad_mask()

        self.output_cfa = self.apply_cfa()

        if self.cfa == 'quad_bayer' and self.binning:
            self.binning_matrix = self.get_quad_binning_matrix()
            self.output_binning = self.apply_binning()
            self.output_subsampling = self.apply_subsampling()
            self.output = self.output_subsampling

        else:
            self.output = self.output_cfa


    def apply_cfa(self):
        return increase_dimensions(sum(reduce_dimensions(self.input)[:, k] * reduce_dimensions(self.cfa_mask)[:, k] for k in range(self.N_k)), (self.N_i, self.N_j))


    def apply_binning(self):
        res = np.zeros((self.N_i, self.N_j))

        for i in range(self.N_i):
            for j in range(self.N_j):
                l = int(self.binning_matrix[i, j])
                if l != 0:
                    l_i = ((i + l) >= self.N_i) * (self.N_i - i - l) + l
                    l_j = ((j + l) >= self.N_j) * (self.N_j - j - l) + l
                    
                    mean_value = np.mean(self.output_cfa[i:i + l_i, j:j + l_j])

                    for ii in range(l_i):
                        for jj in range(l_j):
                            res[i + ii, j + jj] = mean_value

        return res


    def apply_subsampling(self):
        l_GCD = np.gcd.reduce(reduce_dimensions(self.binning_matrix).astype('int32'))
        P_i, P_j = int(np.ceil(self.N_i / l_GCD)), int(np.ceil(self.N_j / l_GCD))

        res = np.zeros((P_i, P_j))

        for i in range(P_i):
            for j in range(P_j):
                res[i, j] = self.output_binning[i * l_GCD, j * l_GCD]

        return res


    def get_bayer_mask(self):
        mask = np.zeros_like(self.input)

        for i in range(self.N_i):
            for j in range(self.N_j):
                for k in range(self.N_k):
                    if k == self.k_r:
                        if i % 2 == 0 and j % 2 == 1:
                            mask[i, j, k] = 1

                    elif k == self.k_g:
                        if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                            mask[i, j, k] = 1

                    elif k == self.k_b:
                        if i % 2 == 1 and j % 2 == 0:
                            mask[i, j, k] = 1

        return mask


    def get_quad_mask(self):
        mask = np.zeros_like(self.input)

        for i in range(self.N_i):
            for j in range(self.N_j):
                for k in range(self.N_k):
                    if k == self.k_r:
                        if i % 4 < 2 and j % 4 >= 2:
                            mask[i, j, k] = 1

                    elif k == self.k_g:
                        if (i % 4 < 2 and j % 4 < 2) or (i % 4 >= 2 and j % 4 >= 2):
                            mask[i, j, k] = 1

                    elif k == self.k_b:
                        if i % 4 >= 2 and j % 4 < 2:
                            mask[i, j, k] = 1

        return mask


    def get_quad_binning_matrix(self):
        res = np.zeros((self.N_i, self.N_j))

        for i in range(self.N_i):
            for j in range(self.N_j):
                if i % 2 == 0 and j % 2 == 0:
                    res[i, j] = 2

        return res


    def get_parameters(self):
        return self.N_i, self.N_j, np.gcd.reduce(reduce_dimensions(self.binning_matrix).astype('int32'))


    def save_results(self):
        create_output_dirs()
        output_dir = path.join('output', 'forward_model_outputs')
        input_name_without_extension = path.basename(path.splitext(self.input_name)[0])

        if not self.binning:
            plt.imsave(path.join(output_dir, 'output_cfa', input_name_without_extension + '_' + self.cfa + '_CFA.png'), self.output_cfa, cmap='gray')
            plt.imsave(path.join(output_dir, input_name_without_extension + '_' + self.cfa + '.png'), self.output, cmap='gray')

        else:
            plt.imsave(path.join(output_dir, 'output_cfa', input_name_without_extension + '_' + self.cfa + '_CFA.png'), self.output_cfa, cmap='gray')
            plt.imsave(path.join(output_dir, 'output_binning', input_name_without_extension + '_' + self.cfa + '_binned.png'), self.output_binning, cmap='gray')
            plt.imsave(path.join(output_dir, 'output_subsampling', input_name_without_extension + '_' + self.cfa + '_subsampled.png'), self.output_subsampling, cmap='gray')
            plt.imsave(path.join(output_dir, input_name_without_extension + '_binned_' + self.cfa + '.png'), self.output, cmap='gray')