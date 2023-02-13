import random
from typing import Optional, List, Any, Tuple

from domain.entitites import AliveEntity, PrayNoBrain
from domain.objects import Movement, Coordinates, PrayFood
from contrib.utils import logger


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

    @property
    def game_over(self) -> bool:
        for y, row in enumerate(self.matrix):
            for x, entity in enumerate(row):
                if isinstance(entity, AliveEntity):
                    return False
        return True

    def setup_initial_state(self, live_objs: List[AliveEntity], pray_foods: int, nutrition=3):
        for live_obj in live_objs:
            self._set_object_randomly(live_obj)

        for pray_food in range(pray_foods):
            self._set_object_randomly(PrayFood(nutrition=nutrition))

    def step_forward(self) -> Tuple[List[List], bool]:
        next_state = self._get_next_state()
        return next_state, self.game_over

    def _make_move(self, movement: Movement, obj: AliveEntity) -> None:
        from_: Optional[Coordinates] = self._get_object_coordinates(obj)
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

    def _respawn_object(self, where: Coordinates, obj) -> None:
        if self.matrix[where.y][where.x] == 0:
            self.matrix[where.y][where.x] = obj
        else:
            raise NotVacantPlaceException('Desired position != 0')

    def _erase_object(self, where: Coordinates) -> None:
        self.matrix[where.y][where.x] = 0

    def _is_empty_coordinates(self, where: Coordinates) -> bool:
        return True if self.matrix[where.y][where.x] == 0 else False

    def _get_object_coordinates(self, obj) -> Optional[Coordinates]:
        for y, row in enumerate(self.matrix):
            for x, element in enumerate(row):
                if element == obj:
                    return Coordinates(x, y)

    def _get_next_state(self) -> List[List]:
        moved_entity_cash = []

        for y, row in enumerate(self.matrix):
            for x, entity in enumerate(row):
                if isinstance(entity, AliveEntity):

                    if entity.health == 0:
                        self._erase_object(Coordinates(x, y))
                        logger.debug(f'Object {entity} died! Lived for: {entity.lived_for}')
                        continue

                    if entity in moved_entity_cash:
                        continue

                    movement: Movement = entity.get_move()
                    self._make_move(movement, entity)
                    moved_entity_cash.append(entity)

        return self.matrix

    def _get_random_coordinates(self) -> Coordinates:
        return Coordinates(
            random.randint(1, self.width) - 1,
            random.randint(1, self.height) - 1,
        )

    def _set_object_randomly(self, obj: Any) -> None:
        if not self.has_space_left:
            raise SetupEnvironmentError('No space left in environment')

        in_process = True

        while in_process:
            random_coordinates: Coordinates = self._get_random_coordinates()
            if self._is_empty_coordinates(random_coordinates):
                self._respawn_object(random_coordinates, obj)
                in_process = False

    def __repr__(self):
        return f'Matrix {self.width}x{self.height}'


if __name__ == '__main__':
    # benchmark
    results_lived_for = []
    for episode in range(20):

        # setup
        environment = Environment(10, 10)
        pray = PrayNoBrain('Mammoth', 5)
        environment.setup_initial_state(live_objs=[pray], pray_foods=10)

        # run
        game_over = False
        while not game_over:
            _, game_over = environment.step_forward()
        results_lived_for.append(pray.lived_for)

    logger.info(f'Average lived for is: {sum(results_lived_for)/len(results_lived_for)}')
