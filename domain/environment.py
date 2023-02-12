import random
from typing import Optional, List, Any

from domain.entitites import AliveEntity
from domain.objects import Movement, Coordinates, PrayFood


class EnvironmentException(Exception):
    pass


class NotVacantPlaceException(EnvironmentException):
    """ This place is already occupied """


class UnsupportedMovement(EnvironmentException):
    """ This movement is not supported """


class ObjectNotExistsInEnvironment(EnvironmentException):
    """ Alas """


class SetupEnvironmentError(EnvironmentException):
    """ No space left in environment """


class Environment:

    def __init__(self, width: int, height: int):

        self.width: int = width
        self.height: int = height

        self.matrix: List[List] = [
            [0 if i not in (0, width - 1) and j not in (0, height - 1) else None for j in range(height)]
            for i in range(width)
        ]

    @property
    def has_space_left(self) -> bool:
        for row in self.matrix:
            for place in row:
                if place == 0:
                    return True
        return False

    def make_move(self, movement: Movement, obj: AliveEntity) -> None:
        from_: Optional[Coordinates] = self.get_object_coordinates(obj)
        if not from_:
            raise ObjectNotExistsInEnvironment(f'Object {obj} is missing in environment')

        if movement == Movement.STAY:
            return
        elif movement == Movement.UP:
            desired_coordinates = Coordinates(from_.x, from_.y - 1)
        elif movement == Movement.DOWN:
            desired_coordinates = Coordinates(from_.x, from_.y + 1)
        elif movement == Movement.LEFT:
            desired_coordinates = Coordinates(from_.x - 1, from_.y)
        elif movement == Movement.RIGHT:
            desired_coordinates = Coordinates(from_.x + 1, from_.y)
        elif movement == Movement.UP_LEFT:
            desired_coordinates = Coordinates(from_.x - 1, from_.y - 1)
        elif movement == Movement.UP_RIGHT:
            desired_coordinates = Coordinates(from_.x + 1, from_.y - 1)
        elif movement == Movement.DOWN_LEFT:
            desired_coordinates = Coordinates(from_.x - 1, from_.y + 1)
        elif movement == Movement.DOWN_RIGHT:
            desired_coordinates = Coordinates(from_.x + 1, from_.y + 1)
        else:
            raise UnsupportedMovement(f'This movement is not supported: {movement}')

        if self.matrix[desired_coordinates.y][desired_coordinates.x] == 0:
            self.matrix[desired_coordinates.y][desired_coordinates.x] = obj
            self.matrix[from_.y][from_.x] = 0
        elif isinstance(self.matrix[desired_coordinates.y][desired_coordinates.x], PrayFood):
            obj.eat(self.matrix[desired_coordinates.y][desired_coordinates.x])
            self.matrix[desired_coordinates.y][desired_coordinates.x] = obj
            self.matrix[from_.y][from_.x] = 0

    def respawn_object(self, where: Coordinates, obj) -> None:
        if self.matrix[where.y][where.x] == 0:
            self.matrix[where.y][where.x] = obj
        else:
            raise NotVacantPlaceException('Desired position != 0')

    def erase_object(self, where: Coordinates) -> None:
        self.matrix[where.y][where.x] = 0

    def is_empty(self, where: Coordinates) -> bool:
        return True if self.matrix[where.y][where.x] == 0 else False

    def get_object_coordinates(self, obj) -> Optional[Coordinates]:
        for y, row in enumerate(self.matrix):
            for x, element in enumerate(row):
                if element == obj:
                    return Coordinates(x, y)

    def get_next_state(self) -> List[List]:
        moved_entity_cash = []

        for y, row in enumerate(self.matrix):
            for x, entity in enumerate(row):
                if isinstance(entity, AliveEntity):

                    if entity.health == 0:
                        self.erase_object(Coordinates(x, y))
                        continue

                    if entity in moved_entity_cash:
                        continue

                    movement: Movement = entity.get_move()
                    self.make_move(movement, entity)
                    moved_entity_cash.append(entity)

        return self.matrix

    def __repr__(self):
        return f'Matrix {self.width}x{self.height}'


class EnvironmentRunner:

    def __init__(self, environment: Environment):
        self.environment = environment

    def get_random_coordinates(self) -> Coordinates:
        return Coordinates(
            random.randint(1, self.environment.width)-1,
            random.randint(1, self.environment.height)-1,
        )

    def setup_initial_state(self, live_objs: List[AliveEntity], pray_foods: int):
        for live_obj in live_objs:
            self.set_object_randomly(live_obj)

        for pray_food in range(pray_foods):
            self.set_object_randomly(PrayFood(nutrition=10))

    def set_object_randomly(self, obj: Any) -> None:
        if not self.environment.has_space_left:
            raise SetupEnvironmentError('No space left in environment')

        in_process = True

        while in_process:
            random_coordinates: Coordinates = self.get_random_coordinates()
            if self.environment.is_empty(random_coordinates):
                self.environment.respawn_object(random_coordinates, obj)
                in_process = False

    def run_live(self):
        pass
