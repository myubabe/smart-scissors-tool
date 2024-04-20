import skimage
import numpy as np
from typing import Sequence, Union
from itertools import product


def unfold(x: np.array, filter_size: Union[int, np.array] = 3):
    feature_size, *spatial = x.shape
    if isinstance(filter_size, int):
        filter_size = np.array((filter_size,) * len(spatial))

    unfolded = np.zeros((feature_size, *filter_size, *spatial))

    def get_spans(shift):
        if shift 