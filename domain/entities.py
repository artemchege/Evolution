from typing import List

import numpy as np

from contrib.utils import logger
from domain.interfaces.entities import AliveEntity

from domain.interfaces.setup import HerbivoreFood


class Herbivore(AliveEntity):
    """ Not trained herbivore, movements are random """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix_converted = HerbivoreMatrixConverter()

    def eat(self, food: HerbivoreFood) -> None:
        self.health += food.nutrition
        logger.debug(f'{self.name} ate! New health: {self.health}')


class Predator(AliveEntity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix_converted = PredatorMatrixConverter()

    def eat(self, food: Herbivore) -> None:
        self.health += food.health
        logger.debug(f'{self.name} ate {food.name}! New health: {self.health}')


class HerbivoreMatrixConverter:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        result = []

        for row in matrix:
            new_row = []
            for element in row:
                if element == 0:
                    new_row.append(1)
                elif element is None or isinstance(element, Herbivore):
                    new_row.append(0)
                elif isinstance(element, HerbivoreFood):
                    new_row.append(2)
                elif isinstance(element, Predator):
                    new_row.append(3)
            result.append(new_row)

        return np.array(result).ravel()


class PredatorMatrixConverter:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        result = []

        for row in matrix:
            new_row = []
            for element in row:
                if element == 0:
                    new_row.append(1)
                elif element is None or isinstance(element, Predator) or isinstance(element, HerbivoreFood):
                    new_row.append(0)
                elif isinstance(element, Herbivore):
                    new_row.append(2)
            result.append(new_row)

        return np.array(result).ravel()
