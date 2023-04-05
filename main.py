import random
from typing import Union, List

import pygame

from contrib.utils import logger
from domain.entities import Herbivore, Predator
from domain.environment import Environment
from domain.utils import StatisticsCollector
from domain.interfaces.setup import Setup
from run_setups import get_setup_for_trained_model_predator_and_herb
from visualization.visualize import Visualizer


class Runner:
    def __init__(
            self,
            setup: Setup,
    ):
        self.setup: Setup = setup
        self.environment = Environment(
            window_width=setup.window.width,
            window_height=self.setup.window.height,
            sustain_services=self.setup.sustain_services,
        )
        self.visualizer: Visualizer = Visualizer(self.environment)
        self.statistics_collector = StatisticsCollector(environment=self.environment, filename='stat')

    def run(self):
        entities: List[Union[Predator, Herbivore]] = [
            entity_setup.entity_type(
                health=entity_setup.initial_health,
                name=f"{entity_setup.entity_type.__name__}#{random.randint(1, 10000)}",
                brain=entity_setup.brain(),
                birth_config=entity_setup.birth,
            )
            for entity_setup in self.setup.entities
            for _ in range(entity_setup.entities_amount)
        ]

        self.environment.setup_initial_state(entities=entities)

        run = True
        while run:
            state_to_render, _ = self.environment.step_living_regime()
            self.visualizer.render_step(state_to_render)
            self.statistics_collector.make_snapshot()

            if self.setup.cycle_length and self.environment.cycle >= self.setup.cycle_length:
                run = False

            if self.environment.game_over:
                run = False

        pygame.quit()
        self.statistics_collector.dump_to_file()
        logger.info(f'Game was closed {self.environment.cycle=}')


if __name__ == '__main__':
    Runner(setup=get_setup_for_trained_model_predator_and_herb()).run()
    # train_best_herbivore()
    # train_best_predator()
