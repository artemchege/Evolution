import pathlib
import random
from typing import Protocol, Tuple

from stable_baselines3 import PPO

from domain.exceptions import UnknownObservationSpace
from domain.objects import Movement, ObservationRange


class Brain(Protocol):
    """ An interface to brain, stable baseline 3 model have the same pair of methods """

    def get_copy(self):
        pass

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        pass

    def set_next_movement(self, movement: int):
        pass

    def required_observation_range(self) -> ObservationRange:
        pass


class ControlledBrain:
    """ Brain that return next movement that supposed to be set outside, used with gym Trainer to train models """

    def __init__(self, observation_width: ObservationRange = ObservationRange.ONE_CELL_AROUND):
        self.next_movement = []
        self.observation_range: ObservationRange = observation_width

    @classmethod
    def get_copy(cls):
        return cls()

    def set_next_movement(self, movement: int):
        self.next_movement.append(movement)

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        return self.next_movement.pop(), None

    def required_observation_range(self) -> ObservationRange:
        return self.observation_range


class RandomBrain:
    """ Brain that returns random movements """

    @classmethod
    def get_copy(cls):
        return cls()

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        return random.randint(0, len(Movement) - 1), None

    def set_next_movement(self, movement: int):
        raise NotImplemented('This brain class generates movements itself')

    @staticmethod
    def required_observation_range() -> ObservationRange:
        return ObservationRange.ONE_CELL_AROUND


class TrainedModelMixin:
    @classmethod
    def get_copy(cls):
        return cls()

    def learn(self, *args, **kwargs) -> None:
        self.model.learn(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Tuple:
        return self.model.predict(*args, **kwargs)

    def set_next_movement(self, movement: int):
        raise NotImplemented('This brain class generates movements itself')

    def required_observation_range(self) -> ObservationRange:
        observation_space_length: int = len(self.model.observation_space)
        if observation_space_length == 9:
            return ObservationRange.ONE_CELL_AROUND
        elif observation_space_length == 25:
            return ObservationRange.TWO_CELL_AROUND
        else:
            raise UnknownObservationSpace(f'Cannot match the length: {observation_space_length}')


class TrainedBrainHerbivoreTwoCells100000(TrainedModelMixin):
    model = PPO.load(
        pathlib.Path(__file__).resolve().parent.parent /
        'Training' / 'saved_models' / 'PPO_model_Herbivore_100000_20x20_food60_3_two_cells'
    )


class TrainedBrainHerbivoreOneCells100000(TrainedModelMixin):
    model = PPO.load(
        pathlib.Path(__file__).resolve().parent.parent /
        'Training' / 'saved_models' / 'PPO_model_Herbivore_100000_20x20_food60_3_one_cells'
    )


class TrainedBrainHerbivoreTwoCells1000000(TrainedModelMixin):
    model = PPO.load(
        pathlib.Path(__file__).resolve().parent.parent /
        'Training' / 'saved_models' / 'PPO_model_Herbivore_1000000_20x20_food60_3_two_cells'
    )


class TrainedBrainPredator100000(TrainedModelMixin):
    model = PPO.load(
        pathlib.Path(__file__).resolve().parent.parent /
        'Training' / 'saved_models' / 'PPO_model_Predator_100000_20x20_food30'
    )
