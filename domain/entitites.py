import os
from abc import ABC, abstractmethod
import random
from typing import List

import numpy as np
from stable_baselines3 import PPO

from contrib.utils import logger

from domain.objects import Movement, HerbivoreFood, MOVEMENT_MAPPER_ADJACENT


class MatrixConverter:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        replace_zeroes_to_ones = [[1 if x == 0 else x for x in row] for row in matrix]
        replace_none_to_zeroes = [[0 if x is None or isinstance(x, HerbivoreNoBrain) else x for x in row] for row in
                                  replace_zeroes_to_ones]
        replace_food_to_two = [[2 if isinstance(x, HerbivoreFood) else x for x in row] for row in replace_none_to_zeroes]
        return np.array(replace_food_to_two).ravel()


class AliveEntity(ABC):

    def __init__(self, name, health):
        self.name = name
        self.health = health
        self.lived_for = 0

    @abstractmethod
    def get_move(self, observation: List[List]) -> Movement:
        pass

    @abstractmethod
    def eat(self, food: HerbivoreFood) -> None:
        pass

    def increase_lived_for(self) -> None:
        self.lived_for += 1


class HerbivoreNoBrain(AliveEntity):
    """ Not trained herbivore, movements are random """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self, observation: List[List]) -> Movement:
        self.health -= 1
        self.increase_lived_for()
        next_move = random.choice(list(Movement))
        logger.debug(f'{self} moves {next_move} health {self.health}')
        return next_move

    def eat(self, food: HerbivoreFood) -> None:
        self.health += food.nutrition
        logger.debug(f'{self.name} ate! New health: {self.health}')

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class HerbivoreTrained100000(HerbivoreNoBrain):
    """ Prev trained and saved model """

    matrix_converted = MatrixConverter()
    model_path = os.path.join('Training', 'saved_models', 'PPO_model_herbivore_100000_8x8_food50_5')
    model = PPO.load(model_path)

    def get_move(self, observation: List[List]) -> Movement:
        self.health -= 1
        self.increase_lived_for()

        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.model.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')
        return movement
