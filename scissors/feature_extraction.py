
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
    'distance_value': 3,
    'history_capacity': 16,
    'n_image_values': 255,
    'n_magnitude_values': 255,

    # other params
    'maximum_cost': 255,
    'snap_scale': 3
}


class StaticExtractor:
    def __init__(self, laplace_kernels=None, laplace_weights=None, std=None,
                 laplace_w=None, direction_w=None, maximum_cost=None):
        """
        Class for computing static features.

        Parameters
        ----------
        laplace_kernels : Sequence[int]
            defines the size of the laplace kernels.
        std : int
            standard deviation for gaussian kernel.
        laplace_weights : Sequence[float]
            defines strength of different laplace filters
        laplace_w : float
            weight of laplace zero-crossing  features
        direction_w : float
            weight of gradient direction features
        maximum_cost : int
           specifies the largest possible integer cost
        """

        std = std or default_params['gaussian_kernel']
        laplace_w = laplace_w or default_params['laplace']
        direction_w = direction_w or default_params['direction']
        maximum_cost = maximum_cost or default_params['maximum_cost']

        self.std = std
        self.maximum_cost = maximum_cost