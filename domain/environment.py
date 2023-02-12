from typing import Optional

from domain.entitites import AliveEntity
from domain.objects import Movement, Coordinates


class EnvironmentException(Exception):
    pass


class NotVacantPlaceException(EnvironmentException):
    """ This place is already occupied """


class UnsupportedMovement(EnvironmentException):
    """ This movement is not supported """


class ObjectNotExistsInEnvironment(EnvironmentException):
    """ Alas """


class Environment:

    def __init__(self, width: int, height: int):

        self.width = width
        self.height = height

        self.matrix = [
            [0 if i not in (0, width - 1) and j not in (0, height - 1) else None for j in range(height)]
            for i in range(width)
        ]

    def make_move(self, movement: Movement, obj: AliveEntity):
        frm: Optional[Coordinates] = self.get_object_coordinate(obj)
        if not frm:
            raise ObjectNotExistsInEnvironment(f'Object {obj} is missing in environment')

        if movement == Movement.STAY:
            return
        elif movement == Movement.UP:
            desired_position = Coordinates(frm.x, frm.y - 1)
        elif movement == Movement.DOWN:
            desired_position = Coordinates(frm.x, frm.y + 1)
        elif movement == Movement.LEFT:
            desired_position = Coordinates(frm.x - 1, frm.y)
        elif movement == Movement.RIGHT:
            desired_position = Coordinates(frm.x + 1, frm.y)
        elif movement == Movement.UP_LEFT:
            desired_position = Coordinates(frm.x - 1, frm.y - 1)
        elif movement == Movement.UP_RIGHT:
            desired_position = Coordinates(frm.x + 1, frm.y - 1)
        elif movement == Movement.DOWN_LEFT:
            desired_position = Coordinates(frm.x - 1, frm.y + 1)
        elif movement == Movement.DOWN_RIGHT:
            desired_position = Coordinates(frm.x + 1, frm.y + 1)
        else:
            raise UnsupportedMovement(f'This movement is not supported: {movement}')

        if self.matrix[desired_position.y][desired_position.x] == 0:
            self.matrix[desired_position.y][desired_position.x] = obj  # noqa
            self.matrix[frm.y][frm.x] = 0

    def respawn_object(self, where: Coordinates, obj):
        if self.matrix[where.y][where.x] == 0:
            self.matrix[where.y][where.x] = obj
        else:
            raise NotVacantPlaceException('Desired position != 0')

    def get_object_coordinate(self, obj) -> Optional[Coordinates]:
        for y, row in enumerate(self.matrix):
            for x, element in enumerate(row):
                if element == obj:
                    return Coordinates(x, y)

    def get_next_state(self):
        for y, row in enumerate(self.matrix):
            for x, entity in enumerate(row):
                if isinstance(entity, AliveEntity):
                    movement: Movement = entity.get_move()
                    self.make_move(movement, entity)
        return self.matrix

    def __repr__(self):
        return f'Matrix {self.width}x{self.height}'
