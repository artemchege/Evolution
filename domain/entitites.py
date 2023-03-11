import os
from abc import ABC, abstractmethod
import random
from typing import List

import numpy as np
from stable_baselines3 import PPO

from contrib.utils import logger

from domain.objects import Movement, HerbivoreFood, MOVEMENT_MAPPER_ADJACENT, TrainingSetup


class MatrixConverter:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        # TODO: оптимизировать и сделать все за один цикл

        replace_zeroes_to_ones = [[1 if x == 0 else x for x in row] for row in matrix]
        replace_none_to_zeroes = [[0 if x is None or isinstance(x, HerbivoreBase) else x for x in row] for row in
                                  replace_zeroes_to_ones]
        replace_food_to_two = [[2 if isinstance(x, HerbivoreFood) else x for x in row] for row in replace_none_to_zeroes]
        return np.array(replace_food_to_two).ravel()


class AliveEntity(ABC):

    def __init__(self, name, health, brain=None):
        self.name = name
        self.health = health
        self.lived_for = 0
        self.brain = brain

    @abstractmethod
    def get_move(self, observation: List[List]) -> Movement:
        # TODO: вынести сюда
        # self.health -= 1
        # self.increase_lived_for()
        pass

    @abstractmethod
    def eat(self, food: HerbivoreFood) -> None:
        pass

    def increase_lived_for(self) -> None:
        self.lived_for += 1


class HerbivoreBase(AliveEntity):
    """ Not trained herbivore, movements are random """

    # TODO: отрефачить, сделать так чтоб брейн был у всех, только у бейз будет брейн который на предикт ворачивает
    #  рандом и тогда не надо будет одинаково переопределять get_move у всех подклассов, а также по дефолту вызывать
    #  дообучение

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix_converted = MatrixConverter()

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


class HerbivoreTrained100000(HerbivoreBase):
    """ Prev trained and saved model, 100_000 cycles, smart enough to live forever """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.brain = PPO.load(
            os.path.join('Training', 'saved_models', 'PPO_model_herbivore_100000_8x8_food50_5')
        )

    def get_move(self, observation: List[List]) -> Movement:
        self.health -= 1
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
        self.brain = PPO("MlpPolicy", gym, verbose=1, tensorboard_log=None)

    def get_move(self, observation: List[List]) -> Movement:
        self.health -= 1
        self.increase_lived_for()

        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')

        # TODO: вызвать дообучение
        self.brain.learn(total_timesteps=1)

        return movement

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

