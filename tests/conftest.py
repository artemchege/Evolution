import pytest

from domain.entitites import HerbivoreTrain, Herbivore
from domain.environment import Environment
from domain.objects import Setup, WindowSetup, FoodSetup, HerbivoreSetup, HerbivoreTrainSetup


@pytest.fixture
def basic_setup() -> Setup:
    return Setup(
            window=WindowSetup(
                width=10,
                height=10,
            ),
            food=FoodSetup(
                herbivore_food_amount=3,
                herbivore_food_nutrition=3,
                replenish_food=False,
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
                herbivore_trainer_class=Herbivore,
                max_live_training_length=5000,
            )
        )


@pytest.fixture
def basic_env(basic_setup) -> Environment:
    return Environment(setup=basic_setup)


@pytest.fixture
def basic_herbivore():
    return Herbivore(
        name='Test herbivore',
        health=10,
    )
