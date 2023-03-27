import random
import timeit
import pygame
from stable_baselines3 import PPO

from contrib.utils import logger
from domain.entitites import HerbivoreBase, HerbivoreTrain, RandomBrain
from domain.environment import Environment
from domain.objects import Setup, WindowSetup, FoodSetup, HerbivoreSetup, HerbivoreTrainSetup, Movement
from evolution.training import HerbivoreTrainer
from visualization.visualize import Visualizer


class Runner:
    def __init__(
            self,
            setup: Setup,
    ):
        self.setup: Setup = setup
        self.environment = Environment(
            setup=self.setup,
        )
        self.visualizer: Visualizer = Visualizer(self.environment)

    def run(self):
        herbivores = [
            self.setup.herbivore.herbivore_class(
                environment=self.environment,
                health=self.setup.herbivore.herbivore_initial_health,
                name=f"Initial herbivore #{random.randint(1, 10000)}",
            ) for _ in range(self.setup.herbivore.herbivores_amount)
        ]
        self.environment.setup_initial_state(herbivores=herbivores)

        run = True
        while run:
            state_to_render, _ = self.environment.step_living_regime()
            self.visualizer.render_step(state_to_render)

        pygame.quit()
        logger.debug('Game was closed')


def go_runner():
    Runner(
        setup=Setup(
            window=WindowSetup(
                width=16,
                height=16,
            ),
            food=FoodSetup(
                herbivore_food_amount=50,
                herbivore_food_nutrition=3,
                replenish_food=True,
            ),
            herbivore=HerbivoreSetup(
                herbivores_amount=5,
                herbivore_class=HerbivoreTrain,
                herbivore_initial_health=20,
                birth_after=None,
                learn_frequency=2,
                learn_n_steps=512,
            ),
            train=HerbivoreTrainSetup(
                herbivore_trainer_class=HerbivoreBase,
                max_live_training_length=5000,
            )
        ),
    ).run()


def create_trained_model():
    setup = Setup(
        window=WindowSetup(
            width=16,
            height=16,
        ),
        food=FoodSetup(
            herbivore_food_amount=30,
            herbivore_food_nutrition=3,
            replenish_food=True,
        ),
        herbivore=HerbivoreSetup(
            herbivores_amount=1,
            herbivore_class=HerbivoreBase,
            herbivore_initial_health=20,
            birth_after=None,
            learn_frequency=2,
            learn_n_steps=512,
        ),
        train=HerbivoreTrainSetup(
            herbivore_trainer_class=HerbivoreBase,
            max_live_training_length=5000,
        )
    )

    trainer = HerbivoreTrainer(
        movement_class=Movement,
        environment=Environment(
            setup=setup,
        ),
        setup=setup,
        visualizer=None,
    )

    model = PPO("MlpPolicy", trainer, verbose=1, tensorboard_log=None)
    model.learn(total_timesteps=100)
    return model


def run_environment_in_range():
    setup = Setup(
        window=WindowSetup(
            width=16,
            height=16,
        ),
        food=FoodSetup(
            herbivore_food_amount=50,
            herbivore_food_nutrition=3,
            replenish_food=True,
        ),
        herbivore=HerbivoreSetup(
            herbivores_amount=5,
            herbivore_class=HerbivoreTrain,
            herbivore_initial_health=20,
            birth_after=None,
            learn_frequency=2,
            learn_n_steps=512,
        ),
        train=HerbivoreTrainSetup(
            herbivore_trainer_class=HerbivoreBase,
            max_live_training_length=5000,
        )
    )

    environment = Environment(setup)
    herbivores = [
        HerbivoreBase(
            health=100000000,
            name=f"Initial herbivore #{random.randint(1, 10000)}",
            brain=RandomBrain(),
        ) for _ in range(50)
    ]
    environment.setup_initial_state(herbivores=herbivores)

    for _ in range(1000):
        environment.step_living_regime()
        # print(environment.matrix)


def benchmark_efficiency_training():
    num = 500
    execution_time = timeit.timeit(create_trained_model, number=num)
    print("Время выполнения функции (AVG): ", execution_time/num)


def benchmark_just_environment():
    num = 10
    execution_time = timeit.timeit(run_environment_in_range, number=num)
    print("Время выполнения функции (AVG): ", execution_time / num)


if __name__ == '__main__':
    # go_runner()
    # benchmark_efficiency_training()
    benchmark_just_environment()
    # create_trained_model()
