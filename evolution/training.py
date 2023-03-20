from enum import EnumMeta
from typing import List, Tuple, Optional

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from domain.entitites import HerbivoreBase, MatrixConverter, MatrixConverterV2
from domain.environment import Environment
from domain.objects import Movement, MOVEMENT_MAPPER_ADJACENT, Setup
from visualization.visualize import Visualizer


class HerbivoreTrainer(gym.Env):
    """ Custom Gym environment that runs training process """

    def __init__(
            self,
            movement_class: EnumMeta,
            environment: Environment,
            setup: Setup,
            visualizer: Optional[Visualizer] = None,
    ):
        self.environment: Environment = environment
        self.action_space = Discrete(len(movement_class))
        self.observation_space = MultiDiscrete([3] * 9)
        self.setup: Setup = setup
        self.herbivore = None
        self.visualizer = visualizer  # Visualizer(self.environment)
        self.matrix_converted = MatrixConverter()
        # self.matrix_converted = MatrixConverterV2()

        # TODO: проверить что травоядное с брейном нужнго типа

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[action]

        previous_health: int = self.herbivore.health
        self.herbivore.brain.set_next_movement(movement)
        _, herbivore_died = self.environment.step_living_regime()
        current_health: int = self.herbivore.health
        reward: int = 1 if current_health > previous_health else 0
        done: bool = (
            True if herbivore_died or self.herbivore.lived_for >= self.setup.train.max_live_training_length else False
        )

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
        #  травоядных и еды, тренировать на идентичном окружении, требуется более глубокая доработка с объединением
        #  enviromnets Live/Training в одно, чтоб когда мы тренировали сущность, то она тренировалась с заданным
        #  количеством хищников и травоядных, причем эти хищники и травоядные отвечали за свои действия сами, то есть
        #  Live режим в train режиме
        self.environment.setup_initial_state(herbivores=[self.herbivore])

        return self._get_herbivore_observation()

    def _get_herbivore_observation(self) -> np.ndarray:
        state_around_obj_list: List[List] = self.environment.get_living_object_observation(self.herbivore)
        return self.matrix_converted.from_environment_to_stable_baseline(state_around_obj_list)

    def _create_herbivore(self) -> HerbivoreBase:
        return self.setup.train.herbivore_trainer_class(
            name="Background trainer entity",
            health=self.setup.herbivore.herbivore_initial_health,
        )
