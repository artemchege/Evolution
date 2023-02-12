from abc import ABC, abstractmethod

from domain.objects import Movement, Coordinates


class AliveEntity(ABC):

    def __init__(self, name, health, environment):
        self.name = name
        self.health = health
        self.environment = environment
        self.x = 0
        self.y = 0

    @abstractmethod
    def move(self, where: Movement):
        pass

    @abstractmethod
    def eat(self):
        pass

    @abstractmethod
    def change_coordinates(self, to: Coordinates):
        pass

    @property
    def get_coordinates(self) -> Coordinates:
        return Coordinates(self.x, self.y)


class Pray(AliveEntity):

    # возможно ханить положение в пространстве не отвественность существа

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def move(self, current_coordinates):
        pass

    def if_can_move(self) -> bool:
        pass

    def eat(self):
        pass

    def change_coordinates(self, to: Coordinates):
        self.x, self.y = to.x, to.y

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class Predator:
    pass
