from abc import ABC, abstractmethod
import random
from contrib.utils import logger

from domain.objects import Movement, PrayFood


class AliveEntity(ABC):

    def __init__(self, name, health):
        self.name = name
        self.health = health
        self.lived_for = 0

    @abstractmethod
    def get_move(self, environment_around=None) -> Movement:
        pass

    @abstractmethod
    def eat(self, food: PrayFood) -> None:
        pass

    def increase_lived_for(self) -> None:
        self.lived_for += 1


class PrayNoBrain(AliveEntity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self, environment_around=None) -> Movement:
        self.health -= 1
        self.increase_lived_for()
        next_move = random.choice(list(Movement))
        logger.debug(f'{self} moves {next_move} health {self.health}')
        return next_move

    def eat(self, food: PrayFood) -> None:
        self.health += food.nutrition
        logger.debug(f'{self.name} ate! New health: {self.health}')

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class Predator:
    pass
