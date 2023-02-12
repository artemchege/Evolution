from abc import ABC, abstractmethod


class AliveEntity(ABC):

    def __init__(self, name, health):
        self.name = name
        self.health = health

    @abstractmethod
    def get_move(self):
        pass

    @abstractmethod
    def eat(self):
        pass


class Pray(AliveEntity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_move(self):
        pass

    def eat(self):
        pass

    def __repr__(self):
        return f'Pray {self.name}, health: {self.health}'


class Predator:
    pass
