import random
from copy import copy
from typing import Optional, List, Any, Tuple, Dict

from domain.entitites import AliveEntity
from domain.exceptions import NotVacantPlaceException, UnsupportedMovement, ObjectNotExistsInEnvironment, \
    SetupEnvironmentError
from domain.objects import Movement, Coordinates, HerbivoreFood, Setup
from contrib.utils import logger


class Environment:
    """ Environment that represent world around living objects and key rules """

    def __init__(self, setup: Setup):
        self.setup: Setup = setup

        self.width: int = setup.window.width
        self.height: int = setup.window.height
        self.replenish_food: bool = setup.food.replenish_food
        self.food_nutrition: int = setup.food.herbivore_food_nutrition
        self.herbivore_food_amount = setup.food.herbivore_food_amount
        self.matrix: List[List] = self._create_blank_matrix()

        # Место для новой абстракции хранения
        self.alive_entities_coords: Dict[AliveEntity, Coordinates] = {}

        self.cycle: int = 0

    @property
    def has_space_left(self) -> bool:
        for row in self.matrix:
            for place in row:
                if place == 0:
                    return True
        return False

    @property
    def game_over(self) -> bool:
        return True if len(self.alive_entities_coords) == 0 else False

    def increment_cycle(self):
        self.cycle += 1

    def setup_initial_state(self, herbivores: List[AliveEntity]) -> None:
        self.matrix = self._create_blank_matrix()
        self.alive_entities_coords = {}

        if len(herbivores) < 1:
            raise SetupEnvironmentError("No herbivores were provided")

        for herbivore in herbivores:
            self._set_object_randomly(herbivore)

        for _ in range(self.herbivore_food_amount):
            self._set_object_randomly(HerbivoreFood(nutrition=self.food_nutrition))

    def get_living_object_observation(self, living_obj: AliveEntity) -> List[List]:
        return self._get_observation(self.alive_entities_coords[living_obj])

    def step_living_regime(self) -> Tuple[List[List], bool]:
        self.increment_cycle()
        next_state: List[List] = self._get_next_state()
        return next_state, self.game_over

    def _get_next_state(self) -> List[List]:
        do_not_move: List[AliveEntity] = []

        for entity in copy(self.alive_entities_coords):

            if entity in do_not_move:
                continue

            if self.setup.birth.birth_after:
                if child := entity.give_birth():
                    self._set_obj_near(near=self.alive_entities_coords[entity], obj=child)
                    do_not_move.append(child)

            observation: List[List] = self._get_observation(self.alive_entities_coords[entity])
            movement: Movement = entity.get_move(observation=observation)
            self._make_move(movement, entity)
            do_not_move.append(entity)

        self._erase_dead_entities()
        return self.matrix

    def _create_blank_matrix(self):
        return [
            [0 if i not in (0, self.width - 1) and j not in (0, self.height - 1) else None for j in range(self.height)]
            for i in range(self.width)
        ]

    def _set_object_randomly(self, obj: Any) -> None:
        if not self.has_space_left:
            raise SetupEnvironmentError('No space left in environment')

        in_process = True

        while in_process:
            random_coordinates: Coordinates = self._get_random_coordinates()
            if self._is_empty_coordinates(random_coordinates):
                self._respawn_object(random_coordinates, obj)
                in_process = False

    def _is_empty_coordinates(self, where: Coordinates) -> bool:
        return True if self.matrix[where.y][where.x] == 0 else False

    def _respawn_object(self, where: Coordinates, obj: Any) -> None:
        if self.matrix[where.y][where.x] == 0:
            self.matrix[where.y][where.x] = obj
            if isinstance(obj, AliveEntity):
                self._change_coordinates_of_alive_object(obj, where)
            logger.debug(f'Object {obj} was respawned at {where}')
        else:
            raise NotVacantPlaceException('Desired position != 0')

    def _get_random_coordinates(self) -> Coordinates:
        return Coordinates(
            random.randint(1, self.width) - 1,
            random.randint(1, self.height) - 1,
        )

    def _get_observation(self, point_of_observation: Coordinates) -> List[List]:
        return [
            row[point_of_observation.x - 1:point_of_observation.x + 2]
            for row in self.matrix[point_of_observation.y - 1:point_of_observation.y + 2]
        ]

    def _get_object_coordinates(self, obj: AliveEntity) -> Optional[Coordinates]:
        if obj not in self.alive_entities_coords:
            raise ObjectNotExistsInEnvironment(f'Object {obj} is missing in environment')
        return self.alive_entities_coords[obj]

    def _change_coordinates_of_alive_object(self, entity: AliveEntity, new_coordinates: Coordinates) -> None:
        self.alive_entities_coords[entity] = new_coordinates

    def _set_obj_near(self, near: Coordinates, obj: Any) -> None:
        coordinates_around: List[Coordinates] = [
            Coordinates(near.x + x, near.y + y) for y in range(-1, 2) for x in range(-1, 2)
        ]

        for coordinate in coordinates_around:
            if self._is_empty_coordinates(coordinate):
                self._respawn_object(coordinate, obj)
                return

        self._set_object_randomly(obj)
        logger.warning('Cannot respawn near to the parent, respawning randomly')

    def _erase_object(self, obj: Any, where: Coordinates) -> None:
        self.matrix[where.y][where.x] = 0
        if isinstance(obj, AliveEntity):
            if obj in self.alive_entities_coords:
                del self.alive_entities_coords[obj]
            else:
                raise ObjectNotExistsInEnvironment(f'Error while deleting object: {obj}, where: {where}')

    def _erase_dead_entities(self):
        dead_entities: List[AliveEntity] = [entity for entity in self.alive_entities_coords if entity.health <= 0]
        for entity in dead_entities:
            self._erase_object(entity, self.alive_entities_coords[entity])

    def _make_move(self, movement: Movement, obj: AliveEntity) -> None:
        from_: Coordinates = self._get_object_coordinates(obj)

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
            self._change_coordinates_of_alive_object(obj, desired_coordinates)
        elif isinstance(self.matrix[desired_coordinates.y][desired_coordinates.x], HerbivoreFood):
            obj.eat(self.matrix[desired_coordinates.y][desired_coordinates.x])
            self.matrix[desired_coordinates.y][desired_coordinates.x] = obj
            self.matrix[from_.y][from_.x] = 0
            if self.replenish_food:
                self._set_object_randomly(HerbivoreFood(self.food_nutrition))
            self._change_coordinates_of_alive_object(obj, desired_coordinates)
