from abc import ABC, abstractmethod
import random
from contrib.utils import logger

from domain.objects import Movement, PrayFood


class AliveEntity(ABC):

    def __init__(self, name, health):
        self.name = name
        self.health = health

    @abstractmethod
    def get_move(self, environment_around=None) -> Movement:
        pass

    @abstractmethod
    def eat(self, food: PrayFood):
        pass


class Pray(AliveEntity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self, environment_around=None) -> Movement:

        # TODO: Reinforcement Neural Network

        self.health -= 1
        next_move = random.choice(list(Movement))
        logger.debug(f'Pray {self} moves {next_move} health {self.health}')
        return next_move

    def eat(self, food: PrayFood):
        self.health += food.nutrition
        logger.debug(f'Pray {self.name} ate! New health: {self.health}')

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class Predator:
    pass
