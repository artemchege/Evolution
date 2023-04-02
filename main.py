import random
from functools import partial
from typing import Union, List

import pygame
from stable_baselines3 import PPO

from contrib.utils import logger
from domain.entitites import Herbivore, Predator
from domain.brain import TrainedBrain100000, RandomBrain
from domain.environment import Environment, StatisticsCollector
from domain.sustain_service import (
    HerbivoreFoodSustainEvery3CycleService,
    HerbivoreFoodSustainEveryCycleService,
    HerbivoreFoodSustainConstantService,
    HerbivoreSustainConstantService
)
from domain.objects import Setup, WindowSetup, EntitySetup, TrainSetup, Movement, BirthSetup
from evolution.training import PredatorTrainer, BrainForTraining, HerbivoreTrainer
from visualization.visualize import Visualizer


def get_setup_for_trained_model_herb():
    return Setup(
        window=WindowSetup(
            width=50,
            height=50,
        ),
        sustain_services=[
            HerbivoreFoodSustainEvery3CycleService(
                initial_food_amount=300, food_nutrition=15,
            )
        ],
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=5,
                brain=partial(TrainedBrain100000),
                initial_health=10,
                birth=BirthSetup(
                    decrease_health_after_birth=250,
                    health_after_birth=10,
                    birth_after=300,
                ),
            ),
        ]
    )


def get_setup_for_trained_model_predator():
    return Setup(
        window=WindowSetup(
            width=50,
            height=50,
        ),
        sustain_services=[
            HerbivoreFoodSustainEvery3CycleService(
                initial_food_amount=300, food_nutrition=15,
            ),
            # HerbivoreFoodSustainConstantService(required_amount_of_herb_food=200, food_nutrition=3),
            HerbivoreSustainConstantService(
                required_amount_of_herbivores=30, initial_herbivore_health=10,
            ),
        ],
        entities=[
            EntitySetup(
                entity_type=Predator,
                entities_amount=1,
                initial_health=1000,
                brain=RandomBrain,
                birth=BirthSetup(
                    decrease_health_after_birth=250,
                    health_after_birth=10,
                    birth_after=300,
                ),
            ),
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=20,
                initial_health=30,
                brain=RandomBrain,
                birth=None,
            ),
        ]
    )


def setup_for_real_time_training_visualization_herb_evolving():
    window_setup = WindowSetup(
        width=50, height=50,
    )
    basic_herbivore_food_service = HerbivoreFoodSustainConstantService(
        required_amount_of_herb_food=500, food_nutrition=3,
    )

    herb_brain = partial(
        BrainForTraining,
        train_setup=TrainSetup(
            learn_frequency=2,
            learn_timesteps=1000,
            learn_n_steps=512,
        ),
        gym_trainer=HerbivoreTrainer(
            movement_class=Movement,
            environment=Environment(
                window_width=window_setup.width,
                window_height=window_setup.height,
                sustain_services=[basic_herbivore_food_service],
            ),
            max_live_training_length=3000,
            health_after_birth=20,
        )
    )

    return Setup(
        window=window_setup,
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=5,
                brain=herb_brain,
                birth=BirthSetup(
                    decrease_health_after_birth=10,
                    health_after_birth=10,
                    birth_after=15,
                ),
                initial_health=10,
            ),
        ],
        sustain_services=[HerbivoreFoodSustainEvery3CycleService(food_nutrition=3, initial_food_amount=500)],
    )


def setup_for_real_time_training_visualization_predators_evolving():
    window_setup = WindowSetup(
        width=50, height=50,
    )

    predator_brain = partial(
        BrainForTraining,
        train_setup=TrainSetup(
            learn_frequency=2,
            learn_timesteps=1000,
            learn_n_steps=512,
        ),
        gym_trainer=PredatorTrainer(
            movement_class=Movement,
            environment=Environment(
                window_width=window_setup.width,
                window_height=window_setup.height,
                sustain_services=[
                    HerbivoreSustainConstantService(required_amount_of_herbivores=100, initial_herbivore_health=10,),
                ],
            ),
            max_live_training_length=3000,
            health_after_birth=20,
        )
    )

    return Setup(
        window=window_setup,
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=5,
                brain=predator_brain,
                birth=BirthSetup(
                    decrease_health_after_birth=10,
                    health_after_birth=10,
                    birth_after=15,
                ),
                initial_health=10,
            ),
        ],
        sustain_services=[
            HerbivoreSustainConstantService(required_amount_of_herbivores=100, initial_herbivore_health=10,),
        ],
    )


class Runner:
    def __init__(
            self,
            setup: Setup,
    ):
        self.setup: Setup = setup
        self.environment = Environment(
            window_width=setup.window.width, window_height=self.setup.window.height,
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
        logger.info('Game was closed')


def train_best_herbivore():
    gym_trainer = HerbivoreTrainer(
        movement_class=Movement,
        environment=Environment(
            window_width=50,
            window_height=50,
            sustain_services=[HerbivoreFoodSustainEvery3CycleService(
                initial_food_amount=600, food_nutrition=10,
            )],
        ),
        max_live_training_length=3000,
        health_after_birth=20,
    )
    model = PPO(
        "MlpPolicy", gym_trainer, verbose=1, tensorboard_log=None,
    )
    model.learn(total_timesteps=100000, progress_bar=True)


def train_best_predator():
    gym_trainer = PredatorTrainer(
        movement_class=Movement,
        environment=Environment(
            window_width=20,
            window_height=20,
            sustain_services=[HerbivoreSustainConstantService(
                    required_amount_of_herbivores=30, initial_herbivore_health=10,
                )
            ],
        ),
        max_live_training_length=3000,
        health_after_birth=20,
    )
    model = PPO(
        "MlpPolicy", gym_trainer, verbose=1, tensorboard_log=None,
    )
    model.learn(total_timesteps=100000, progress_bar=True)


if __name__ == '__main__':
    # Runner(setup=setup_for_real_time_training_visualization_herb_evolving()).run()
    # train_best_herbivore()
    train_best_predator()
