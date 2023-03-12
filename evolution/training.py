from enum import EnumMeta
from typing import List, Tuple, Optional

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from domain.entitites import HerbivoreBase, MatrixConverter
from domain.environment import EnvironmentTrainRegime
from domain.objects import Movement, MOVEMENT_MAPPER_ADJACENT, TrainingSetup
from visualization.visualize import Visualizer


class HerbivoreGym(gym.Env):
    """ Custom Gym environment that runs training process """

    def __init__(
            self,
            movement_class: EnumMeta,
            environment: EnvironmentTrainRegime,  # noqa
            setup: TrainingSetup,
            visualizer=Optional[Visualizer],
    ):
        self.environment: EnvironmentTrainRegime = environment
        self.action_space = Discrete(len(movement_class))
        self.observation_space = MultiDiscrete([3] * 9)
        self.setup: TrainingSetup = setup
        self.herbivore = None
        self.visualizer = visualizer  # Visualizer(self.environment)
        self.matrix_converted = MatrixConverter()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[action]

        previous_health: int = self.herbivore.health
        self.environment.set_next_movement(movement)
        _, herbivore_died = self.environment.step_living_regime()
        current_health: int = self.herbivore.health
        reward: int = 1 if current_health > previous_health else 0
        done: bool = True if herbivore_died or self.herbivore.lived_for >= self.setup.live_length else False

        if not done:
            observation: np.ndarray = self._get_herbivore_observation()
        else:
            observation: np.ndarray = np.array([[0 for _ in range(3)] for _ in range(3)]).ravel()

        return observation, reward, done, {}

    def render(self, mode="human"):
        if mode == 'human' and self.visualizer:
            self.visualizer.render_step(self.environment.matrix)

    def reset(self) -> np.ndarray:
        self.herbivore: HerbivoreBase = self._create_herbivore()

        # TODO: сделать так чтоб сюда копировалось текущее состояние env с тем же количеством хищников и
        #  травоядных и еды, тренировать на идентичном окружении
        self.environment.setup_initial_state(
            live_objs=[self.herbivore],
        )

        return self._get_herbivore_observation()

    def _get_herbivore_observation(self) -> np.ndarray:
        state_around_obj_list: List[List] = self.environment.get_living_object_observation(self.herbivore)
        return self.matrix_converted.from_environment_to_stable_baseline(state_around_obj_list)

    def _create_herbivore(self) -> HerbivoreBase:
        return self.setup.living_object_class(
            name=self.setup.living_object_name,
            health=self.setup.living_object_initial_health,
        )
