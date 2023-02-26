import random
from abc import abstractmethod, ABC
from typing import Optional, List, Any, Tuple

from domain.entitites import AliveEntity
from domain.objects import Movement, Coordinates, HerbivoreFood
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


class Environment(ABC):
    """ Environment that represent world around living objects and key rules """

    def __init__(self, width: int, height: int, food_nutrition: int, replenish_food: bool = True):

        self.width: int = width
        self.height: int = height
        self.replenish_food: bool = replenish_food
        self.food_nutrition: int = food_nutrition
        self.matrix: List[List] = self._create_blank_matrix()

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

    def setup_initial_state(self, live_objs: List[AliveEntity], herbivore_food_amount: int, nutrition=3):
        self.matrix = self._create_blank_matrix()

        for live_obj in live_objs:
            self._set_object_randomly(live_obj)

        for _ in range(herbivore_food_amount):
            self._set_object_randomly(HerbivoreFood(nutrition=nutrition))

    def step_living_regime(self) -> Tuple[List[List], bool]:
        next_state: List[List] = self._get_next_state()
        return next_state, self.game_over

    def get_living_object_observation(self, living_obj: AliveEntity) -> List[List]:
        living_object_coordinates: Optional[Coordinates] = self._get_object_coordinates(living_obj)
        return self._get_observation(living_object_coordinates)

    @abstractmethod
    def _get_next_state(self) -> List[List]:
        pass

    def _create_blank_matrix(self):
        return [
            [0 if i not in (0, self.width - 1) and j not in (0, self.height - 1) else None for j in range(self.height)]
            for i in range(self.width)
        ]

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
        elif isinstance(self.matrix[desired_coordinates.y][desired_coordinates.x], HerbivoreFood):
            obj.eat(self.matrix[desired_coordinates.y][desired_coordinates.x])
            self.matrix[desired_coordinates.y][desired_coordinates.x] = obj
            self.matrix[from_.y][from_.x] = 0
            if self.replenish_food:
                self._set_object_randomly(HerbivoreFood(self.food_nutrition))

    def _respawn_object(self, where: Coordinates, obj) -> None:
        if self.matrix[where.y][where.x] == 0:
            self.matrix[where.y][where.x] = obj
            logger.debug(f'Object {obj} was respawned at {where}')
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

    def _get_observation(self, point_of_observation: Coordinates) -> List[List]:
        return [
            row[point_of_observation.x - 1:point_of_observation.x + 2]
            for row in self.matrix[point_of_observation.y - 1:point_of_observation.y + 2]
        ]

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


class EnvironmentTrainRegime(Environment):
    """ Must be used in gym.Env environment runners to train models. Key difference is that in this subclass an
    action of a living object is set from outside (from training model) in process of RL training. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.movement_from_outside: Optional[Movement] = None

    def set_next_movement(self, movement: Movement):
        self.movement_from_outside = movement

    def _get_next_state(self) -> List[List]:
        moved_entity_cash: List[AliveEntity] = []

        for y, row in enumerate(self.matrix):
            for x, entity in enumerate(row):
                if isinstance(entity, AliveEntity):

                    if entity.health == 0:
                        self._erase_object(Coordinates(x, y))
                        logger.debug(f'Object {entity} died! Lived for: {entity.lived_for}')
                        continue

                    if entity in moved_entity_cash:
                        continue

                    entity.get_move(observation=[[]])  # does not matter, movement is set in set_movement_from_outside
                    movement: Movement = self.movement_from_outside

                    self._make_move(movement, entity)
                    moved_entity_cash.append(entity)

        return self.matrix


class EnvironmentLiveRegime(Environment):
    """ Must be used with already trained entities without gym.Env just to look at how objects behaves. Key difference
    is that in this subclass each living object id asked about its next movement """

    def _get_next_state(self) -> List[List]:
        moved_entity_cash: List[AliveEntity] = []

        for y, row in enumerate(self.matrix):
            for x, entity in enumerate(row):
                if isinstance(entity, AliveEntity):

                    if entity.health == 0:
                        self._erase_object(Coordinates(x, y))
                        logger.debug(f'Object {entity} died! Lived for: {entity.lived_for}')
                        continue

                    if entity in moved_entity_cash:
                        continue

                    observation: List[List] = self._get_observation(Coordinates(x, y))
                    movement: Movement = entity.get_move(observation=observation)  # ask each entity about next move
                    self._make_move(movement, entity)
                    moved_entity_cash.append(entity)

        return self.matrix
