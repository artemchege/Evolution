from functools import partial

import pytest

from domain.brain import RandomBrain, ControlledBrain
from domain.entitites import Herbivore, Predator
from domain.environment import Environment
from domain.objects import Setup, WindowSetup, FoodSetup, HerbivoreSetup, HerbivoreTrainSetup, BirthSetup


@pytest.fixture
def basic_setup() -> Setup:
    return Setup(
        window=WindowSetup(
            width=16,
            height=16,
        ),
        food=FoodSetup(
            herbivore_food_amount=50,
            herbivore_food_nutrition=3,
            replenish_food=None,
        ),
        herbivore=HerbivoreSetup(
            herbivores_amount=5,
            brain=partial(RandomBrain),
        ),
        train=HerbivoreTrainSetup(),
        birth=BirthSetup(
            decrease_health_after_birth=10,
            health_after_birth=10,
            birth_after=None,
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
        brain=ControlledBrain(),
    )


@pytest.fixture
def basic_predator():
    return Predator(
        name='Test predator',
        health=10,
        brain=RandomBrain(),
    )
