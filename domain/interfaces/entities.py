import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from contrib.utils import logger
from domain.exceptions import InvalidEntityState
from domain.interfaces.brain import Brain
from domain.interfaces.objects import ObservationRange, Movement, MOVEMENT_MAPPER_ADJACENT


@dataclass(frozen=True)
class BirthSetup:
    decrease_health_after_birth: int
    health_after_birth: int
    birth_after: int


class AliveEntity(ABC):
    """ Alive entity that lives in an environment """

    def __init__(
            self,
            name: str,
            health: int,
            brain: Brain,
            birth_config: Optional[BirthSetup] = None,
    ):
        """ Expects a name, initial health, brain reference and birth config if an entity is going to be replicate
        itself """

        self.name = name
        self.health = health
        self.lived_for = 0
        self.birth_config: BirthSetup = birth_config
        self.brain: Brain = brain
        self.matrix_converted: MatrixConverter = None  # noqa
        self.uid = uuid.uuid4()
        self.eaten: bool = False

    @property
    def is_dead(self) -> bool:
        """ Predicate of whether entity is dead or no """

        return True if self.health <= 0 or self.eaten else False

    def increase_lived_for(self) -> None:
        """ Increase counter of lived for """

        self.lived_for += 1

    def increase_health(self, amount: int):
        """ Increase amount of health """

        self.health += amount

    def decrease_health(self, amount: int):
        """ Decrease amount of health """

        self.health -= amount
        if self.health < 0:
            raise InvalidEntityState("Health is below 0")

    def was_eaten(self):
        """ Entity was eaten by another entity """

        self.eaten = True

    @abstractmethod
    def eat(self, food) -> None:
        """ Eat environment object if eatable """
        pass

    def give_birth(self) -> Optional['AliveEntity']:
        """ Replicate itself if birth_config is specified. Bequeath the current state of development to the child  """

        if self.birth_config and self.health > self.birth_config.birth_after:
            child = self.__class__(
                name=f'Child-{random.randint(1, 1000)}',
                health=self.birth_config.health_after_birth,
                brain=self.brain.get_copy(),
                birth_config=self.birth_config,
            )
            self.decrease_health(self.birth_config.decrease_health_after_birth)
            return child

    def get_move(self, observation: List[List]) -> Movement:
        """ Return the next movement. Based on the environment observation """

        self.decrease_health(1)
        self.increase_lived_for()
        converted_observation = self.matrix_converted.from_environment_to_stable_baseline(observation)
        action_num, _ = self.brain.predict(converted_observation)
        movement: Movement = MOVEMENT_MAPPER_ADJACENT[int(action_num)]
        logger.debug(f'{self} moves {movement} health {self.health}')
        return movement

    def get_observation_range(self) -> ObservationRange:
        """ Get the observation range with which the brain was trained (1x, 2x) """

        return self.brain.required_observation_range()

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, AliveEntity):
            return self.uid == other.uid
        return NotImplemented

    def __repr__(self):
        return f'{self.name}, health: {self.health}'

