import random
from enum import EnumMeta
from typing import List, Tuple, Optional

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from stable_baselines3 import PPO

from contrib.utils import logger
from domain.brain import ControlledBrain
from domain.entitites import Herbivore, HerbivoreMatrixConverter
from domain.environment import Environment
from domain.objects import TrainSetup
from visualization.visualize import Visualizer


class EntityTrainer(gym.Env):
    """ Custom Gym environment that runs training process """

    def __init__(
            self,
            movement_class: EnumMeta,
            environment: Environment,
            max_live_training_length: int,
            health_after_birth: int,
            visualizer: Optional[Visualizer] = None,
    ):
        self.environment: Environment = environment
        self.action_space = Discrete(len(movement_class))
        self.observation_space = MultiDiscrete([3] * 9)
        self.entity = None
        self.visualizer = visualizer
        self.matrix_converted = HerbivoreMatrixConverter()
        self.max_live_training_length: int = max_live_training_length
        self.health_after_birth: int = health_after_birth

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        previous_health: int = self.entity.health
        self.entity.brain.set_next_movement(action)
        _, game_over = self.environment.step_living_regime()
        current_health: int = self.entity.health
        reward: int = 1 if current_health > previous_health else 0
        done: bool = (
            True
            if game_over
            or self.entity.eaten
            or self.entity.lived_for >= self.max_live_training_length
            else False
        )

        if not done:
            observation: np.ndarray = self._get_entity_observation()
        else:
            observation: np.ndarray = np.array([[0 for _ in range(3)] for _ in range(3)]).ravel()

        return observation, reward, done, {}

    def render(self, mode="human"):
        if mode == 'human' and self.visualizer:
            self.visualizer.render_step(self.environment.matrix)

    def reset(self) -> np.ndarray:
        self.entity = Herbivore(
            name="Background trainer entity",
            health=self.health_after_birth,
            brain=ControlledBrain(),
            birth_config=None,
        )
        self.environment.setup_initial_state([self.entity])
        return self._get_entity_observation()

    def _get_entity_observation(self) -> np.ndarray:
        state_around_obj_list: List[List] = self.environment.get_living_object_observation(self.entity)
        return self.matrix_converted.from_environment_to_stable_baseline(state_around_obj_list)


class BrainForTraining:
    def __init__(
            self, train_setup: TrainSetup, gym_trainer: EntityTrainer
    ):
        self.train_setup: TrainSetup = train_setup
        self.gym_trainer: EntityTrainer = gym_trainer
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
