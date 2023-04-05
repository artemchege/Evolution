from typing import Protocol, Tuple

from domain.interfaces.setup import ObservationRange


class Brain(Protocol):
    """ An interface to brain, stable baseline 3 model have the same pair of methods """

    def get_copy(self):
        pass

    def learn(self, *args, **kwargs) -> None:
        pass

    def predict(self, *args, **kwargs) -> Tuple:
        pass

    def set_next_movement(self, movement: int):
        pass

    def required_observation_range(self) -> ObservationRange:
        pass
