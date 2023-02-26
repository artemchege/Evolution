import enum
from dataclasses import dataclass
from typing import Callable


@dataclass
class Coordinates:
    x: int
    y: int


class Movement(enum.Enum):
    STAY = 'STAY'
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    UP_LEFT = 'UP_LEFT'
    UP_RIGHT = 'UP_RIGHT'
    DOWN_LEFT = 'DOWN_LEFT'
    DOWN_RIGHT = 'DOWN_RIGHT'


MOVEMENT_MAPPER_ADJACENT = {
    0: Movement.STAY,
    1: Movement.UP_LEFT,
    2: Movement.UP,
    3: Movement.UP_RIGHT,
    4: Movement.RIGHT,
    5: Movement.DOWN_RIGHT,
    6: Movement.DOWN,
    7: Movement.DOWN_LEFT,
    8: Movement.LEFT,
}


@dataclass
class HerbivoreFood:
    nutrition: int


@dataclass
class TrainingSetup:
    # food setup
    herbivore_food_amount: int
    herbivore_food_nutrition: int
    replenish_food: bool

    # living obj setup
    living_object_name: str
    living_object_class: Callable  # HerbivoreNoBrain, cannot typehint due to import circle
    living_object_initial_health: int
    live_length: int = 1000
