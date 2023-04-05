from dataclasses import dataclass
from typing import Optional, List, Type, Callable

from domain.interfaces.brain import Brain
from domain.interfaces.entities import BirthSetup, AliveEntity
from domain.interfaces.environment import SustainEnvironmentService


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
class EntitySetup:
    entity_type: Type[AliveEntity]
    entities_amount: int
    initial_health: int
    brain: Callable[[Brain], None]
    birth: Optional[BirthSetup]


@dataclass(frozen=True)
class Setup:
    window: WindowSetup
    sustain_services:  List[SustainEnvironmentService]
    entities: List[EntitySetup]
    cycle_length: Optional[int] = None
