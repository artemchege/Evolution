import os
import random
from typing import Protocol, Tuple

from stable_baselines3 import PPO

from domain.objects import Movement


class Brain(Protocol):
    """ An interface to brain, stable baseline 3 model have the same pair of methods """

    def get_copy(self):
        pass

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        pass


class ControlledBrain:
    """ Brain that return next movement that supposed to be set outside, used with gym Trainer to train models """

    def __init__(self):
        self.next_movement = []

    @classmethod
    def get_copy(cls):
        return cls()

    def set_next_movement(self, movement: Movement):
        self.next_movement.append(movement)

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        return self.next_movement.pop(), None


class RandomBrain:
    """ Brain that returns random movements """

    @classmethod
    def get_copy(cls):
        return cls()

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        return random.choice(list(Movement)), None


class TrainedBrain100000:
    model = PPO.load(os.path.join('Training', 'saved_models', 'PPO_model_herbivore_100000_8x8_food50_5'))

    @classmethod
    def get_copy(cls):
        return cls()

    def learn(self, *args, **kwargs) -> None:
        self.model.learn(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Tuple:
        return self.model.predict(*args, **kwargs)
