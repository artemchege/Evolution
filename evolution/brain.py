import pathlib
import random
from typing import Tuple

import gym
from stable_baselines3 import PPO

from contrib.utils import logger
from domain.interfaces.setup import TrainSetup
from domain.interfaces.objects import ObservationRange, Movement

from domain.exceptions import UnknownObservationSpace


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
    """ Previously trained brain, stable_baseline model """

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


def get_user_trained_brain(model_name: str) -> TrainedModelMixin:
    brain = TrainedModelMixin()
    brain.model = PPO.load(
        pathlib.Path(__file__).resolve().parent.parent /
        'Training' / 'saved_models' / model_name
    )
    return brain


class BrainForTraining:
    def __init__(
            self, train_setup: TrainSetup, gym_trainer: gym.Env
    ):
        self.train_setup: TrainSetup = train_setup
        self.gym_trainer: gym.Env = gym_trainer
        self.model = PPO(
            "MlpPolicy", self.gym_trainer, verbose=1, tensorboard_log=None, n_steps=self.train_setup.learn_n_steps,
        )

    def predict(self, *args, **kwargs) -> Tuple:
        if random.randint(0, self.train_setup.learn_frequency) == 0:
            logger.debug(f"Brain {id(self)} started learning")
            self.learn(total_timesteps=self.train_setup.learn_timesteps)
        return self.model.predict(*args, **kwargs)

    def learn(self, *args, **kwargs):
        return self.model.learn(*args, **kwargs)

    def get_copy(self):
        brain = self.__class__(
            train_setup=self.train_setup,
            gym_trainer=self.gym_trainer,
        )
        brain.model.set_parameters(self.model.get_parameters())
        return brain

    def required_observation_range(self) -> ObservationRange:
        observation_space_length: int = len(self.model.observation_space)
        if observation_space_length == 9:
            return ObservationRange.ONE_CELL_AROUND
        elif observation_space_length == 25:
            return ObservationRange.TWO_CELL_AROUND
        else:
            raise UnknownObservationSpace(f'Cannot match the length: {observation_space_length}')
