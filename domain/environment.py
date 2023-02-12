from domain.entitites import AliveEntity
from domain.objects import Movement, Coordinates


class EnvironmentException(Exception):
    pass


class NotVacantPlaceException(EnvironmentException):
    """ This place is already occupied """


class Environment:

    def __init__(self, width: int, height: int):

        self.width = width
        self.height = height

        self.matrix = [
            [0 if i not in (0, width - 1) and j not in (0, height - 1) else None for j in range(height)]
            for i in range(width)
        ]

    def make_move(self, frm: Coordinates, movement: Movement, who: AliveEntity):
        if movement == Movement.STAY:
            desired_position = Coordinates(frm.x, frm.y)
        elif movement == Movement.UP:
            desired_position = Coordinates(frm.x, frm.y - 1)
        elif movement == Movement.DOWN:
            desired_position = Coordinates(frm.x, frm.y + 1)
        elif movement == Movement.LEFT:
            desired_position = Coordinates(frm.x - 1, frm.y)
        else:
            # right
            desired_position = Coordinates(frm.x + 1, frm.y)

        if self.matrix[desired_position.y][desired_position.x] == 0:
            self.matrix[desired_position.y][desired_position.x] = who  # noqa
            self.matrix[frm.y][frm.x] = 0
            who.change_coordinates(desired_position)

    def respawn_object(self, where: Coordinates, obj):
        if self.matrix[where.y][where.x] == 0:
            self.matrix[where.y][where.x] = obj
            obj.change_coordinates(where)
        else:
            raise NotVacantPlaceException('desired position != 0')

    def __repr__(self):
        return f'Matrix {self.width}x{self.height}'
