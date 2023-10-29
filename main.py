import argparse
import random
from typing import List

import pygame

from contrib.utils import logger
from domain.entities import EntityType
from domain.environment import Environment
from domain.interfaces.entities import AliveEntity
from domain.interfaces.objects import ObservationRange
from domain.utils import StatisticsCollector
from domain.interfaces.setup import Setup
from run_setups import get_setup_for_trained_model_predator_and_herb, train_the_best_entity
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
        entities: List[AliveEntity] = [
            entity_setup.entity_type(
                health=entity_setup.initial_health,
                name=f"{entity_setup.entity_type.__name__}#{random.randint(1, 10000)}",
                brain=entity_setup.brain(),  # noqa
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
    # Runner(setup=get_setup_for_trained_model_predator_and_herb()).run()

    parser = argparse.ArgumentParser()
    command_parser = parser.add_subparsers(dest='command')

    train_the_best_model = command_parser.add_parser('train_the_best_model', help='Train the best model of given type')
    train_the_best_model.add_argument(
        '--width', type=int, metavar='WIDTH', required=True, help='Width of the field',
    )
    train_the_best_model.add_argument(
        '--height', type=int, metavar='HEIGHT', required=True, help='Height of the field',
    )
    train_the_best_model.add_argument(
        '--entity_type', choices=['predator', 'herbivore'], required=True, help='Type of entity to train',
    )
    train_the_best_model.add_argument(
        '--health_after_birth', type=int, metavar='HEALTH', required=True, help='Health after birth',
    )
    train_the_best_model.add_argument(
        '--observation_range', choices=['one_cell_around', 'two_cell_around'], required=True,
        help='Observation range',
    )
    train_the_best_model.add_argument(
        '--total_timesteps', type=int, metavar='TOTAL_TIMESTEPS', required=True,
        help='Amount of timestamps for training',
    )
    train_the_best_model.add_argument(
        '--path_for_saving', type=str, metavar='PATH_FOR_SAVING', required=True,
        help='Path for saving the model',
    )

    args = parser.parse_args()

    if args.command == 'train_the_best_model':
        train_the_best_entity(
            window_width=args.width,
            window_height=args.height,
            max_live_training_length=1000,
            health_after_birth=args.health_after_birth,
            total_timestep=args.total_timesteps,
            observation_range=(
                ObservationRange.ONE_CELL_AROUND if args.observation_range == 'one_cell_around'
                else ObservationRange.TWO_CELL_AROUND
            ),
            entity_type=(
                EntityType.HERBIVORE if args.entity_type == 'herbivore' else EntityType.PREDATOR
            ),
            save_path=args.path_for_saving,
        )
