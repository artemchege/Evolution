import random
import pygame

from contrib.utils import logger
from domain.entitites import HerbivoreBase, HerbivoreTrain
from domain.environment import EnvironmentLiveRegime
from domain.objects import Setup, WindowSetup, FoodSetup, HerbivoreSetup, HerbivoreTrainSetup
from visualization.visualize import Visualizer


class Runner:
    def __init__(
            self,
            setup: Setup,
    ):
        self.setup: Setup = setup

        self.environment = EnvironmentLiveRegime(
            setup=self.setup,
        )
        self.visualizer: Visualizer = Visualizer(self.environment)

    def run(self):
        herbivores = [
            self.setup.herbivore.herbivore_class(
                environment=self.environment,
                health=self.setup.herbivore.herbivore_initial_health,
                name=f"Initial herbivore #{random.randint(1, 10000)}",
                learn_rate_step=self.setup.herbivore.learn_frequency,
                learn_n_steps=self.setup.herbivore.learn_n_steps,
                birth_after=self.setup.herbivore.birth_after,
            ) for _ in range(self.setup.herbivore.herbivores_amount)
        ]
        self.environment.setup_herbivores(herbivores)
        self.environment.setup_initial_state()

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
                herbivore_food_amount=30,
                herbivore_food_nutrition=3,
                replenish_food=True,
            ),
            herbivore=HerbivoreSetup(
                herbivores_amount=1,
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


def main():
    go_runner()


if __name__ == '__main__':
    main()
