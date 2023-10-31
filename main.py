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
from run_setups import (
    train_the_best_entity,
    setup_for_real_time_training_visualization_herb_evolving,
    setup_for_real_time_training_visualization_predator_evolving
)
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

    # TODO: Remove parsers from here to separate file

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

    herbivore_visualization_train_from_scratch = command_parser.add_parser(
        'herbivore_visualization_train_from_scratch',
        help='See how your herbivore is training from scratch, adapt your parameters for the bese survival'
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--width', type=int, metavar='WIDTH', required=True, help='Width of the field', default=50,
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--height', type=int, metavar='HEIGHT', required=True, help='Height of the field', default=50,
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--amount_of_herb_food', type=int, metavar='AMOUNT_OF_HERB_FOOD', required=True, default=35,
        help='Amount of herb food',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--herb_food_nutrition', type=int, metavar='HERB_FOOD_NUTRITION', required=True, default=10,
        help='Nutrition of herb food',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--learning_frequency', type=int, metavar='LEARNING_FREQUENCY', required=True, default=2,
        help='Frequency of learning',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--learning_timesteps', type=int, metavar='LEARNING_TIMESTEPS', required=True, default=1000,
        help='Amount of timesteps for learning',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--learning_n_steps', type=int, metavar='LEARNING_N_STEPS', required=True, default=512,
        help='Amount of steps for learning',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--health_after_birth', type=int, metavar='HEALTH_AFTER_BIRTH', required=True, default=10,
        help='Health after birth',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--observation_range', type=int, metavar='OBSERVATION_RANGE', required=True, default=1,
        help='Observation range',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--start_herb_amount', type=int, metavar='START_HERB_AMOUNT', required=True, default=10,
        help='Start herb amount',
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--decrease_parent_health_after_birth', type=int, metavar='DECREASE_PARENT_HEALTH_AFTER_BIRTH', required=True,
        help='Decrease parent health after birth', default=10,
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--child_health_after_birth', type=int, metavar='CHILD_HEALTH_AFTER_BIRTH', required=True,
        help='Child health after birth', default=10,
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--birth_after_health_amount', type=int, metavar='BIRTH_AFTER_HEALTH_AMOUNT', required=True,
        help='Birth after health amount', default=20,
    )
    herbivore_visualization_train_from_scratch.add_argument(
        '--initial_herb_health', type=int, metavar='INITIAL_HERB_HEALTH', required=True,
        help='Initial herb health', default=10,
    )

    predator_visualization_train_from_scratch = command_parser.add_parser(
        'predator_visualization_train_from_scratch',
        help='See how your predator is training from scratch, adapt your parameters for the bese survival'
    )
    predator_visualization_train_from_scratch.add_argument(
        '--width', type=int, metavar='WIDTH', required=True, help='Width of the field', default=50,
    )
    predator_visualization_train_from_scratch.add_argument(
        '--height', type=int, metavar='HEIGHT', required=True, help='Height of the field', default=50,
    )
    predator_visualization_train_from_scratch.add_argument(
        '--amount_of_pred_food', type=int, metavar='AMOUNT_OF_PRED_FOOD', required=True, default=35,
        help='Amount of predator food',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--pred_food_nutrition', type=int, metavar='PRED_FOOD_NUTRITION', required=True, default=10,
        help='Nutrition of herbivores',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--learning_frequency', type=int, metavar='LEARNING_FREQUENCY', required=True, default=2,
        help='Frequency of learning',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--learning_timesteps', type=int, metavar='LEARNING_TIMESTEPS', required=True, default=1000,
        help='Amount of timesteps for learning',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--learning_n_steps', type=int, metavar='LEARNING_N_STEPS', required=True, default=512,
        help='Amount of steps for learning',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--health_after_birth', type=int, metavar='HEALTH_AFTER_BIRTH', required=True, default=10,
        help='Health after birth',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--observation_range', type=int, metavar='OBSERVATION_RANGE', required=True, default=1,
        help='Observation range',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--start_pred_amount', type=int, metavar='START_PRED_AMOUNT', required=True, default=10,
        help='Start predator amount',
    )
    predator_visualization_train_from_scratch.add_argument(
        '--decrease_parent_health_after_birth', type=int, metavar='DECREASE_PARENT_HEALTH_AFTER_BIRTH', required=True,
        help='Decrease parent health after birth', default=10,
    )
    predator_visualization_train_from_scratch.add_argument(
        '--child_health_after_birth', type=int, metavar='CHILD_HEALTH_AFTER_BIRTH', required=True,
        help='Child health after birth', default=10,
    )
    predator_visualization_train_from_scratch.add_argument(
        '--birth_after_health_amount', type=int, metavar='BIRTH_AFTER_HEALTH_AMOUNT', required=True,
        help='Birth after health amount', default=20,
    )
    predator_visualization_train_from_scratch.add_argument(
        '--initial_pred_health', type=int, metavar='INITIAL_PRED_HEALTH', required=True,
        help='Initial predator health', default=10,
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
    elif args.command == 'herbivore_visualization_train_from_scratch':
        Runner(
            setup=setup_for_real_time_training_visualization_herb_evolving(
                width=args.width,
                height=args.height,
                amount_of_herb_food=args.amount_of_herb_food,
                herb_food_nutrition=args.herb_food_nutrition,
                learning_frequency=args.learning_frequency,
                learning_timesteps=args.learning_timesteps,
                learning_n_steps=args.learning_n_steps,
                health_after_birth=args.health_after_birth,
                observation_range=args.observation_range,
                start_herb_amount=args.start_herb_amount,
                decrease_parent_health_after_birth=args.decrease_parent_health_after_birth,
                child_health_after_birth=args.child_health_after_birth,
                birth_after_health_amount=args.birth_after_health_amount,
                initial_herb_health=args.initial_herb_health,
            )
        ).run()
    elif args.command == 'predator_visualization_train_from_scratch':
        Runner(
            setup=setup_for_real_time_training_visualization_predator_evolving(
                width=args.width,
                height=args.height,
                amount_of_predator_food=args.amount_of_pred_food,
                predator_food_nutrition=args.pred_food_nutrition,
                learning_frequency=args.learning_frequency,
                learning_timesteps=args.learning_timesteps,
                learning_n_steps=args.learning_n_steps,
                health_after_birth=args.health_after_birth,
                observation_range=args.observation_range,
                start_pred_amount=args.start_pred_amount,
                decrease_parent_health_after_birth=args.decrease_parent_health_after_birth,
                child_health_after_birth=args.child_health_after_birth,
                birth_after_health_amount=args.birth_after_health_amount,
                initial_pred_health=args.initial_pred_health,
            )
        ).run()
    else:
        raise ValueError('Unknown command')
