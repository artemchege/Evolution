import pytest

from domain.brain import RandomBrain, ControlledBrain
from domain.entitites import Herbivore, Predator
from domain.environment import Environment


@pytest.fixture
def basic_env() -> Environment:
    return Environment(
        window_width=16,
        window_height=16,
        sustain_services=[],
    )


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
