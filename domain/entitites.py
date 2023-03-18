import os
from abc import ABC, abstractmethod
import random
from typing import List, Protocol, Tuple, Optional

import numpy as np
from stable_baselines3 import PPO

from contrib.utils import logger

from domain.objects import Movement, HerbivoreFood, MOVEMENT_MAPPER_ADJACENT, Setup, HerbivoreSetup


class InvalidEntityState(Exception):
    """ Invalid state """


class MatrixConverter:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        # TODO: можно удалять

        replace_zeroes_to_ones = [[1 if x == 0 else x for x in row] for row in matrix]
        replace_none_to_zeroes = [[0 if x is None or isinstance(x, HerbivoreBase) else x for x in row] for row in
                                  replace_zeroes_to_ones]
        replace_food_to_two = [[2 if isinstance(x, HerbivoreFood) else x for x in row] for row in replace_none_to_zeroes]
        return np.array(replace_food_to_two).ravel()


class MatrixConverterV2:

    @staticmethod
    def from_environment_to_stable_baseline(matrix: List[List]) -> np.ndarray:
        result = []

        for row in matrix:
            new_row = []
            for element in row:
                if element == 0:
                    new_row.append(1)
                elif element is None or isinstance(element, HerbivoreBase):
                    new_row.append(0)
                elif isinstance(element, HerbivoreFood):
                    new_row.append(2)
            result.append(new_row)

        return np.array(result).ravel()


class Brain(Protocol):
    """ An interface to brain, stable baseline 3 model have the same pair of methods """

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        pass


class ControlledBrain:
    """ Brain that return next movement that supposed to be set outside, used with gym Trainer to train models """

    def __init__(self):
        self.next_movement = []

    def set_next_movement(self, movement: Movement):
        self.next_movement.append(movement)

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        return self.next_movement.pop(), None


class AliveEntity(ABC):

    def __init__(self, name: str, health: int):
        self.name = name
        self.health = health
        self.lived_for = 0
        self.brain: Brain = ControlledBrain()
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
        pass

    def __repr__(self):
        return f'{self.name}, health: {self.health}'


class HerbivoreTrained100000(HerbivoreBase):
    """ Prev trained and saved model, 100_000 cycles, smart enough to live forever """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.brain = PPO.load(
            os.path.join('Training', 'saved_models', 'PPO_model_herbivore_100000_8x8_food50_5')
        )

    def give_birth(self) -> Optional['AliveEntity']:
        if self.health > 15:
            child = self.__class__(
                name=f'Child-{random.randint(1,1000)}',
                health=10,
            )
            child.brain = self.brain
            self.decrease_health(10)
            return child

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
    from evolution.training import HerbivoreTrainer

    def __init__(
            self, environment: Environment, birth_after: int, learn_rate_step: int, learn_n_steps: int, *args, **kwargs
    ):
        from evolution.training import HerbivoreTrainer

        super().__init__(*args, **kwargs)

        self.environment = environment
        self.birth_after = birth_after
        self.learn_rate_step = learn_rate_step
        self.learn_n_steps = learn_n_steps

        trainer: HerbivoreTrainer = self._get_trainer()
        self.brain = PPO(
            "MlpPolicy", trainer, verbose=1, tensorboard_log=None, n_steps=self.learn_n_steps
        )

    def get_move(self, observation: List[List]) -> Movement:
        self.decrease_health(1)
        self.increase_lived_for()

        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')

        if random.randint(0, self.learn_rate_step) == 0:
            # self.brain.set_env(self._get_trainer())  # TODO: переинит тренера когда объединю окружения
            self.brain.learn(total_timesteps=1)
            logger.info(f"{self.name} start learning")

        return movement

    def give_birth(self) -> Optional['AliveEntity']:
        if self.health > self.birth_after:
            child = self.__class__(
                name=f'Child-{random.randint(1, 1000)}',
                health=self.environment.setup.herbivore.herbivore_initial_health,
                environment=self.environment,
                learn_rate_step=self.learn_rate_step,
                birth_after=self.birth_after,
                learn_n_steps=self.environment.setup.herbivore.learn_n_steps,
            )
            child.brain.set_parameters(self.brain.get_parameters())  # TODO: проверить работает или нет
            self.decrease_health(10)
            return child

    def _get_trainer(self) -> HerbivoreTrainer:
        from domain.environment import Environment
        from evolution.training import HerbivoreTrainer

        setup = Setup(
            window=self.environment.setup.window,
            food=self.environment.setup.food,
            herbivore=HerbivoreSetup(
                herbivores_amount=1,
                herbivore_class=HerbivoreBase,
                herbivore_initial_health=self.environment.setup.herbivore.herbivore_initial_health,
            ),
            train=self.environment.setup.train,
        )

        return HerbivoreTrainer(
            movement_class=Movement,
            environment=Environment(
                setup=setup,
            ),
            setup=setup,
        )
