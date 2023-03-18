import enum
from dataclasses import dataclass
from typing import Callable, Optional


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
class WindowSetup:
    width: int
    height: int


@dataclass
class FoodSetup:
    herbivore_food_amount: int
    herbivore_food_nutrition: int
    replenish_food: bool


@dataclass
class HerbivoreSetup:
    herbivores_amount: int
    herbivore_class: Callable  # HerbivoreTrain, cannot typehint due to import circle
    herbivore_initial_health: int
    birth_after: Optional[int] = None  # If reproduction is available, if yes, after which amount of heath
    learn_frequency: int = 4  # Randomly after this number of steps
    learn_n_steps: int = 128  # Rollout capacity


@dataclass
class HerbivoreTrainSetup:
    herbivore_trainer_class: Callable  # HerbivoreBase, cannot typehint due to import circle
    max_live_training_length: int = 3000


@dataclass
class Setup:
    window: WindowSetup
    food: FoodSetup
    herbivore: HerbivoreSetup
    train: Optional[HerbivoreTrainSetup]
