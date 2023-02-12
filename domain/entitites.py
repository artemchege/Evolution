from abc import ABC, abstractmethod
import random

from domain.objects import Movement


class AliveEntity(ABC):

    def __init__(self, name, health):
        self.name = name
        self.health = health

    @abstractmethod
    def get_move(self) -> Movement:
        pass

    @abstractmethod
    def eat(self):
        pass


class Pray(AliveEntity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self):

        # TODO: Reinforcement Neural Network

        return random.choice(list(Movement))

    def eat(self):
        pass

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class Predator:
    pass
