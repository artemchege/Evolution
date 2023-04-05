import enum
from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class Coordinates:
    x: int
    y: int


@dataclass(frozen=True)
class HerbivoreFood:
    nutrition: int


@dataclass(frozen=True)
class WindowSetup:
    width: int
    height: int


@dataclass(frozen=True)
class TrainSetup:
    learn_frequency: int = 4  # Randomly after this number of steps
    learn_n_steps: int = 128  # Rollout capacity
    learn_timesteps: int = 1  # noqa


@dataclass(frozen=True)
class BirthSetup:
    decrease_health_after_birth: int
    health_after_birth: int
    birth_after: int


@dataclass(frozen=True)
class EntitySetup:
    entity_type: Any  # Type[Union[Predator, Herbivore]]
    entities_amount: int
    initial_health: int
    brain: Any  # Brain
    birth: Optional[BirthSetup]


@dataclass(frozen=True)
class Setup:
    window: WindowSetup
    sustain_services:  List  # List[SustainEnvironmentService]  # List[SustainService]
    entities: List[EntitySetup]
    cycle_length: Optional[int] = None


class ObservationRange(enum.Enum):
    ONE_CELL_AROUND = 'ONE_CELL_AROUND'
    TWO_CELL_AROUND = 'TWO_CELL_AROUND'


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
