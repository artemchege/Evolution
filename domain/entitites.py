import uuid
from abc import ABC, abstractmethod
import random
from typing import List, Optional

import numpy as np

from contrib.utils import logger
from domain.brain import Brain
from domain.exceptions import InvalidEntityState

from domain.objects import Movement, HerbivoreFood, MOVEMENT_MAPPER_ADJACENT, BirthSetup


class MatrixConverterV2:

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
            result.append(new_row)

        return np.array(result).ravel()


class MatrixConverterDepr:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:

        replace_zeroes_to_ones = [[1 if x == 0 else x for x in row] for row in matrix]
        replace_none_to_zeroes = [[0 if x is None or isinstance(x, Herbivore) else x for x in row] for row in
                                  replace_zeroes_to_ones]
        replace_food_to_two = [[2 if isinstance(x, HerbivoreFood) else x for x in row] for row in replace_none_to_zeroes]
        return np.array(replace_food_to_two).ravel()


class AliveEntity(ABC):

    def __init__(
            self, name: str, health: int, brain: Brain, birth_config: Optional[BirthSetup] = None
    ):
        self.name = name
        self.health = health
        self.lived_for = 0
        self.birth_config: BirthSetup = birth_config
        self.brain: Brain = brain
        self.matrix_converted = MatrixConverterV2()
        self.uid = uuid.uuid4()
        self.eaten: bool = False

    def increase_lived_for(self) -> None:
        self.lived_for += 1

    def increase_health(self, amount: int):
        self.health += amount

    def decrease_health(self, amount: int):
        self.health -= amount
        if self.health < 0:
            raise InvalidEntityState("Health is below 0")

    def was_eaten(self):
        self.eaten = True

    @abstractmethod
    def eat(self, food) -> None:
        pass

    @abstractmethod
    def give_birth(self) -> Optional['AliveEntity']:
        pass

    @abstractmethod
    def get_move(self, observation: List[List]) -> Movement:
        pass

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, AliveEntity):
            return self.uid == other.uid
        return NotImplemented

    def __repr__(self):
        return f'{self.name}, health: {self.health}'


class Herbivore(AliveEntity):
    """ Not trained herbivore, movements are random """

    def eat(self, food: HerbivoreFood) -> None:
        self.health += food.nutrition
        logger.debug(f'{self.name} ate! New health: {self.health}')

    def give_birth(self) -> Optional['AliveEntity']:
        if self.health > self.birth_config.birth_after:
            child = self.__class__(
                name=f'Child-{random.randint(1,1000)}',
                health=self.birth_config.health_after_birth,
                brain=self.brain.get_copy(),
                birth_config=self.birth_config,
            )
            self.decrease_health(self.birth_config.decrease_health_after_birth)
            return child

    def get_move(self, observation: List[List]) -> Movement:
        self.decrease_health(1)
        self.increase_lived_for()
        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')
        return movement


class Predator(AliveEntity):

    def eat(self, food: Herbivore) -> None:
        self.health += food.health
        logger.debug(f'{self.name} ate {food.name}! New health: {self.health}')

    def give_birth(self) -> Optional['AliveEntity']:
        if self.health > self.birth_config.birth_after:
            child = self.__class__(
                name=f'Child-{random.randint(1, 1000)}',
                health=self.birth_config.health_after_birth,
                brain=self.brain.get_copy(),
                birth_config=self.birth_config,
            )
            self.decrease_health(self.birth_config.decrease_health_after_birth)
            return child

    def get_move(self, observation: List[List]) -> Movement:
        self.decrease_health(1)
        self.increase_lived_for()
        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')
        return movement
