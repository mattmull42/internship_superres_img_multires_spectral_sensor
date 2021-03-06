# Forward model and inversion strategies for Quad-Bayer & binning

It is advised to use this code with the internship report for an in-depth explaination.

## Setting of the input

The input considered in the whole project is simply a 3 dimensional tensor whare the 2 first dimensions are the spatial resolutions, and the last dimension is the spectral dimension. Moreover the user must give a spectral stencil, an array of the values of the wavelength corresponding to the channels in the image.

The user can simply call the ```initialize_input(image_path)``` which takes the path to a RGB image (png, jpg) or a netCDF4 archive.

If the user wants to load an image for the inverse problem it is possible to use the ```initialize_inverse_input(image_path)``` in the same fashion.

## Forward operator

To initialize a forward operator the user has to declare an instance of the ```Forward_operator```:
```f = Forward_operator(cfa, input_size, spectral_stencil, binning, noise_level)```, with :
* ```cfa``` being either ```bayer```'' or ```quad-bayer```;
* ```input_size``` the shape of the image;
* ```spectral_stencil``` the array presented above;
* ```noise_level``` the noise to apply to the acquisition.

After that the user can apply the operator to an image like : ```y = f(x)``` where ```x``` is the input image.

Finally the user can save the images in the output folder with : ```f.save_output(operator_name)```.

## Inversion problem

The user has 2 different inversion strategies the sequential and joint approach.

For the first one the class is ```Inverse_problem(cfa, binning, forward_model.get_parameters())``` where ```forward_model.get_parameters()``` are some information about the forward operator.

For the second one the class is ```Inverse_problem_ADMM(cfa, binning, noise_level, input_size, spectral_stencil, niter, sigma, epsilon, box_constraint)```, where :
* ```niter``` the number of iterations of the solver;
* ```sigma``` the proximal step of the g function;
* ```epsilon``` the weight for the regularizer term.

In both cases the user can apply the inversion in the same way as for the forward operator and save the output equally.
