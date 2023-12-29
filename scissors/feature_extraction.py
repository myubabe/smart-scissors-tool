
import numpy as np

from typing import Sequence, Union
from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import gaussian, laplace, sobel_h, sobel_v, sobel

from scissors.search import search
from scissors.utils import unfold, create_spatial_feats, flatten_first_dims, quadratic_kernel, preprocess_image

# Parameters  depend ot the image size.
# These parameters were selected for 512 x 512 images.

default_params = {
    # static params
    # sum of all weights must be equal to 1
    'laplace': 0.3,
    'direction': 0.2,
    'magnitude': 0.2,
    'local': 0.1,
    'inner': 0.1,
    'outer': 0.1,
    'laplace_kernels': [3, 5, 7],
    'gaussian_kernel': 5,
    'laplace_weights': [0.2, 0.3, 0.5],

    # dynamic params
    'hist_std': 2,
    'image_std': 1,