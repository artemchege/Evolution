import random
from functools import partial

import pygame
from stable_baselines3 import PPO

from contrib.utils import logger
from domain.entitites import Herbivore
from domain.brain import TrainedBrain100000
from domain.environment import Environment
from domain.sustain_service import HerbivoreFoodSustainEvery3CycleService, HerbivoreFoodSustainEveryCycleService, \
    HerbivoreFoodSustainConstantService
from domain.objects import Setup, WindowSetup, AliveEntitySetup, TrainSetup, Movement, BirthSetup
from evolution.training import HerbivoreTrainer, BrainForTraining
from visualization.visualize import Visualizer


def get_setup_for_trained_model():
    return Setup(
        window=WindowSetup(
            width=50,
            height=50,
        ),
        sustain_services=[
            HerbivoreFoodSustainEveryCycleService(
                initial_food_amount=600, food_nutrition=30,
            )
        ],
        herbivore=AliveEntitySetup(
            herbivores_amount=5,
            brain=partial(TrainedBrain100000),
            initial_health=10,
            birth=BirthSetup(
                decrease_health_after_birth=250,
                health_after_birth=10,
                birth_after=300,
            ),
        ),
        predator=None,
    )


def get_setup_for_real_time_training_visualization():
    window_setup = WindowSetup(
        width=50, height=50,
    )
    basic_herbivore_food_service = HerbivoreFoodSustainConstantService(
        required_amount_of_herb_food=600, food_nutrition=10,
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
        herbivore=AliveEntitySetup(
            herbivores_amount=5,
            brain=herb_brain,
            birth=BirthSetup(
                decrease_health_after_birth=10,
                health_after_birth=10,
                birth_after=15,
            ),
            initial_health=10,
        ),
        predator=None,
        sustain_services=[HerbivoreFoodSustainEvery3CycleService(food_nutrition=3, initial_food_amount=1000)],
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

    def run(self):
        herbivores = [
            Herbivore(
                health=self.setup.herbivore.initial_health,
                name=f"Initial herbivore #{random.randint(1, 10000)}",
                brain=self.setup.herbivore.brain(),
                birth_config=self.setup.herbivore.birth,
            ) for _ in range(self.setup.herbivore.herbivores_amount)
        ]
        self.environment.setup_initial_state(herbivores=herbivores, predators=[])

        run = True
        while run:
            state_to_render, _ = self.environment.step_living_regime()
            self.visualizer.render_step(state_to_render)

        pygame.quit()
        logger.debug('Game was closed')


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


if __name__ == '__main__':
    Runner(setup=get_setup_for_trained_model()).run()
    # train_best_herbivore()
