import multiprocessing
import os
import copy
from multiprocessing import Process
from abc import ABC, abstractmethod
import random
from typing import List, Protocol, Tuple, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from contrib.utils import logger

from domain.objects import Movement, HerbivoreFood, MOVEMENT_MAPPER_ADJACENT, TrainingSetup


class InvalidEntityState(Exception):
    """ Invalid state """


class MatrixConverter:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        # TODO: оптимизировать и сделать все за один цикл

        replace_zeroes_to_ones = [[1 if x == 0 else x for x in row] for row in matrix]
        replace_none_to_zeroes = [[0 if x is None or isinstance(x, HerbivoreBase) else x for x in row] for row in
                                  replace_zeroes_to_ones]
        replace_food_to_two = [[2 if isinstance(x, HerbivoreFood) else x for x in row] for row in replace_none_to_zeroes]
        return np.array(replace_food_to_two).ravel()


class Brain(Protocol):
    """ An interface to brain, stable baseline 3 model have the same pair of methods """

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        pass


class NoBrain:
    def learn(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs) -> Tuple:  # noqa
        return random.choice(list(Movement)), None


class AliveEntity(ABC):

    def __init__(self, name: str, health: int):
        self.name = name
        self.health = health
        self.lived_for = 0
        self.brain: Brain = NoBrain()
        self.matrix_converted = MatrixConverter()

    def get_move(self, observation: List[List]) -> Movement:
        self.health -= 1
        self.increase_lived_for()
        next_move, _ = self.brain.predict(observation)
        logger.debug(f'{self} moves {next_move} health {self.health}')
        return next_move

    def increase_lived_for(self) -> None:
        self.lived_for += 1

    def increase_health(self, amount: int):
        self.health += amount

    def decrease_health(self, amount: int):
        self.health -= amount
        if self.health < 0:
            raise InvalidEntityState("Health is below 0")

    @abstractmethod
    def eat(self, food: HerbivoreFood) -> None:
        pass

    @abstractmethod
    def give_birth(self) -> Optional['AliveEntity']:
        pass


class HerbivoreBase(AliveEntity):
    """ Not trained herbivore, movements are random """

    def eat(self, food: HerbivoreFood) -> None:
        self.health += food.nutrition
        logger.debug(f'{self.name} ate! New health: {self.health}')

    def give_birth(self) -> Optional['AliveEntity']:
        # TODO: параметризировать, сохранять в родителе init health и сделать его чуток рамндомным +- треть от init
        if self.health > 15:
            child = self.__class__(
                name=f'Child-{random.randint(1,1000)}',
                health=10,
            )
            child.brain = self.brain
            self.decrease_health(10)
            return child

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class HerbivoreTrained100000(HerbivoreBase):
    """ Prev trained and saved model, 100_000 cycles, smart enough to live forever """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.brain = PPO.load(
            os.path.join('Training', 'saved_models', 'PPO_model_herbivore_100000_8x8_food50_5')
        )

    def get_move(self, observation: List[List]) -> Movement:
        self.decrease_health(1)
        self.increase_lived_for()

        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')

        return movement


class HerbivoreTrain(HerbivoreBase):
    """ Not trained model that educate itself after each step """

    # TODO: рефакторинг импортов

    from domain.environment import Environment
    from evolution.training import HerbivoreGym

    def __init__(self, env: Environment, *args, **kwargs):
        from evolution.training import HerbivoreGym

        super().__init__(*args, **kwargs)
        self.env = env

        gym: HerbivoreGym = self._get_gym()

        # gyms_fns = [lambda: self._get_gym() for _ in range(2)]
        # gym = DummyVecEnv(gyms_fns)

        self.brain = PPO("MlpPolicy", gym, verbose=1, tensorboard_log=None, n_steps=1000)

    def get_move(self, observation: List[List]) -> Movement:
        self.decrease_health(1)
        self.increase_lived_for()

        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')

        if random.randint(0, 4) == 0:
            self.brain.learn(total_timesteps=1)
            # p = multiprocessing.Process(target=self.brain.learn, args=(1,))
            # p.start()

        return movement

    def give_birth(self) -> Optional['AliveEntity']:
        # TODO: параметризировать, сохранять в родителе init health и сделать его чуток рамндомным +- треть от init
        if self.health > 15:
            child = self.__class__(
                name=f'Child-{random.randint(1, 1000)}',
                health=10,
                env=self.env,
            )
            child.brain.set_parameters(self.brain.get_parameters())  # TODO: проверить работает или нет
            self.decrease_health(10)
            return child

    def _get_training_setup(self) -> TrainingSetup:
        # TODO: рефакторинг констант и сетапа, сделать так чтоб все числа вынести в одно место, возможно запутался
        return TrainingSetup(
            herbivore_food_amount=self.env.herbivore_food_amount,
            herbivore_food_nutrition=self.env.food_nutrition,
            replenish_food=self.env.replenish_food,
            living_object_name='Mammoth',
            living_object_class=HerbivoreBase,
            living_object_initial_health=10,
            live_length=5000,
        )

    def _get_gym(self) -> HerbivoreGym:
        from domain.environment import EnvironmentTrainRegime
        from evolution.training import HerbivoreGym

        setup: TrainingSetup = self._get_training_setup()
        return HerbivoreGym(
            movement_class=Movement,
            environment=EnvironmentTrainRegime(
                width=self.env.width,
                height=self.env.height,
                replenish_food=setup.replenish_food,
                food_nutrition=setup.herbivore_food_nutrition,
            ),
            setup=setup,
        )

