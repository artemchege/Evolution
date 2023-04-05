from typing import Protocol, List

import numpy as np


class MatrixConverter(Protocol):
    """ Converter that translates matrix state into consumable for training shape """

    def from_environment_to_stable_baseline(self, matrix: List[List]) -> np.ndarray:
        """ From environment to numpy array """
        pass
