from abc import abstractmethod
from typing import List, Dict, Any, Tuple

from domain.interfaces.entities import AliveEntity
from domain.interfaces.objects import Coordinates


class EnvironmentInterface:
    """ Environment that represent world around living objects and key rules. Core domain object """

    def __init__(
            self, window_width: int, window_height: int, sustain_services: List['SustainEnvironmentService'],
    ):
        """ Accept width and height of the environment (int), and services that are responsible for sustaining
        environment in given shape """

        self.width: int = window_width
        self.height: int = window_height
        self.sustain_services: List[SustainEnvironmentService] = sustain_services
        self.matrix: List[List] = self._create_blank_matrix()

        # Place for storage abstraction
        self.alive_entities_coords: Dict[AliveEntity, Coordinates] = {}

        self.cycle: int = 0
        self.herbivore_food_amount: int = 0

    @abstractmethod
    def herbivores_amount(self) -> int:
        """ Amount of herbivores left in the environment """
        pass

    @abstractmethod
    def predators_amount(self) -> int:
        """ Amount of predators left in the environment """
        pass

    @abstractmethod
    def setup_initial_state(self, entities: List[AliveEntity]) -> None:
        """ Set initial objects that start living right from the beginning """
        pass

    @abstractmethod
    def set_object_randomly_in_environment(self, obj: Any) -> None:
        """ Set any objects, that represent environment objects in the environment """
        pass

    @abstractmethod
    def get_living_object_observation(self, living_obj: AliveEntity) -> List[List]:
        """ Get an observation (an environment state) around given alive entity """
        pass

    @abstractmethod
    def step_living_regime(self) -> Tuple[List[List], bool]:
        """ Ask living objects about their next step and change environment state, return new state and boolean wither
        game is finished """
        pass

    @abstractmethod
    def _create_blank_matrix(self) -> List[List]:
        """ Get blank matrix of initialized width and height """
        pass


class SustainEnvironmentService:
    """ Service that watch after some objects in environment and replicate them if needed """

    @abstractmethod
    def initial_sustain(self, environment: EnvironmentInterface) -> None:
        """ Sets objects in an environment that are mandatory to be in the environment from the beginning """
        pass

    @abstractmethod
    def subsequent_sustain(self, environment: EnvironmentInterface) -> None:
        """ Sets objects in an environment that are mandatory to be in the environment from during existence of the
        environment """
        pass
