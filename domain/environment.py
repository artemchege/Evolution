import random
from copy import copy
from typing import Optional, List, Any, Tuple

from domain.entities import Predator, Herbivore
from domain.exceptions import (
    NotVacantPlaceException,
    UnsupportedMovement,
    ObjectNotExistsInEnvironment,
    SetupEnvironmentError,
)
from domain.interfaces.entities import AliveEntity
from domain.interfaces.environment import EnvironmentInterface
from domain.interfaces.setup import HerbivoreFood
from domain.interfaces.objects import Coordinates, ObservationRange, Movement
from contrib.utils import logger


class Environment(EnvironmentInterface):
    """ EnvironmentInterface realization """

    @property
    def has_space_left(self) -> bool:
        for row in self.matrix:
            for place in row:
                if place == 0:
                    return True
        return False

    @property
    def herbivores_amount(self) -> int:
        return len([herbivore for herbivore in self.alive_entities_coords if isinstance(herbivore, Herbivore)])

    @property
    def predators_amount(self) -> int:
        return len([predator for predator in self.alive_entities_coords if isinstance(predator, Predator)])

    @property
    def game_over(self) -> bool:
        return True if len(self.alive_entities_coords) == 0 else False

    def increment_cycle(self):
        self.cycle += 1

    def increment_food_amount(self):
        self.herbivore_food_amount += 1

    def decrease_food_amount(self):
        self.herbivore_food_amount -= 1

    def setup_initial_state(self, entities: List[AliveEntity]) -> None:
        self.matrix = self._create_blank_matrix()

        if len(entities) < 1:
            raise SetupEnvironmentError("No herbivores were provided")

        for entity in entities:
            self.set_object_randomly_in_environment(entity)

        for sustain_service in self.sustain_services:
            sustain_service.initial_sustain(self)

    def set_object_randomly_in_environment(self, obj: Any) -> None:
        if not self.has_space_left:
            raise SetupEnvironmentError('No space left in environment')

        in_process = True

        while in_process:
            random_coordinates: Coordinates = self._get_random_coordinates()
            if self._is_empty_coordinates(random_coordinates):
                self._respawn_object(random_coordinates, obj)
                in_process = False

    def get_living_object_observation(self, living_obj: AliveEntity) -> List[List]:
        observation_range: ObservationRange = living_obj.get_observation_range()
        return (
            self._get_observation_one_cell_around(self.alive_entities_coords[living_obj])
            if observation_range == ObservationRange.ONE_CELL_AROUND
            else self._get_observation_two_cells_around(self.alive_entities_coords[living_obj])
        )

    def step_living_regime(self) -> Tuple[List[List], bool]:
        self.increment_cycle()
        next_state: List[List] = self._get_next_state()
        self._erase_dead_entities()
        for sustain_service in self.sustain_services:
            sustain_service.subsequent_sustain(self)
        return next_state, self.game_over

    def _get_next_state(self) -> List[List]:
        do_not_move: List[AliveEntity] = []

        for entity in copy(self.alive_entities_coords):

            if entity in do_not_move:
                continue

            observation: List[List] = self.get_living_object_observation(entity)
            from_ = self._get_object_coordinates(entity)
            entity_movement: Movement = entity.get_move(observation=observation)
            desired_coordinates: Coordinates = self._movements_to_coordinates(
                movement=entity_movement,
                from_=from_,
            )

            if self.matrix[desired_coordinates.y][desired_coordinates.x] == 0:
                self._change_coordinates_of_alive_object(entity, desired_coordinates, from_=from_)

            if isinstance(entity, Herbivore) and isinstance(
                    self.matrix[desired_coordinates.y][desired_coordinates.x], HerbivoreFood
            ):
                herbivore_food = self.matrix[desired_coordinates.y][desired_coordinates.x]
                entity.eat(herbivore_food)
                self._change_coordinates_of_alive_object(entity, desired_coordinates, from_=from_)
                self.decrease_food_amount()

            if isinstance(entity, Predator) and isinstance(
                    self.matrix[desired_coordinates.y][desired_coordinates.x], Herbivore
            ):
                herbivore = self.matrix[desired_coordinates.y][desired_coordinates.x]
                entity.eat(herbivore)
                herbivore.was_eaten()
                self._erase_object(obj=herbivore, where=self._get_object_coordinates(herbivore))
                self._change_coordinates_of_alive_object(entity, desired_coordinates, from_=from_)
                do_not_move.append(herbivore)
                logger.info(f'{entity} eats {herbivore}')

            if child := entity.give_birth():
                self._set_obj_near(near=self._get_object_coordinates(entity), obj=child)
                do_not_move.append(child)

            do_not_move.append(entity)

        return self.matrix

    def _create_blank_matrix(self) -> List[List]:
        self.herbivore_food_amount = 0
        self.alive_entities_coords = {}
        self.cycle = 0
        return [
            [0 if i not in (0, self.width - 1) and j not in (0, self.height - 1) else None for j in range(self.height)]
            for i in range(self.width)
        ]

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

    def _get_observation_one_cell_around(self, point_of_observation: Coordinates) -> List[List]:
        return [
            row[point_of_observation.x - 1:point_of_observation.x + 2]
            for row in self.matrix[point_of_observation.y - 1:point_of_observation.y + 2]
        ]

    def _get_observation_two_cells_around(self, point_of_observation: Coordinates) -> List[List]:
        return [
            [
                self.matrix[i][j] if 0 <= i < len(self.matrix) and 0 <= j < len(self.matrix[0]) else None
                for j in range(point_of_observation.x - 2, point_of_observation.x + 3)
            ]
            if 0 <= i < len(self.matrix)
            else [None for _ in range(point_of_observation.x - 2, point_of_observation.x + 3)]
            for i in range(point_of_observation.y - 2, point_of_observation.y + 3)
        ]

    def _get_object_coordinates(self, obj: AliveEntity) -> Optional[Coordinates]:
        if obj not in self.alive_entities_coords:
            raise ObjectNotExistsInEnvironment(f'Object {obj} is missing in environment')
        return self.alive_entities_coords[obj]

    def _change_coordinates_of_alive_object(
            self, entity: AliveEntity, new_coordinates: Coordinates, from_: Optional[Coordinates] = None
    ) -> None:
        self.alive_entities_coords[entity] = new_coordinates
        self.matrix[new_coordinates.y][new_coordinates.x] = entity
        if from_:
            self.matrix[from_.y][from_.x] = 0

    def _set_obj_near(self, near: Coordinates, obj: Any) -> None:
        coordinates_around: List[Coordinates] = [
            Coordinates(near.x + x, near.y + y) for y in range(-1, 2) for x in range(-1, 2)
        ]

        for coordinate in coordinates_around:
            if self._is_empty_coordinates(coordinate):
                self._respawn_object(coordinate, obj)
                return

        self.set_object_randomly_in_environment(obj)
        logger.warning('Cannot respawn near to the parent, respawning randomly')

    def _erase_object(self, obj: AliveEntity, where: Coordinates) -> None:
        self.matrix[where.y][where.x] = 0
        if isinstance(obj, AliveEntity):
            if obj in self.alive_entities_coords:
                del self.alive_entities_coords[obj]
            else:
                raise ObjectNotExistsInEnvironment(f'Error while deleting object: {obj}, where: {where}')

    def _erase_dead_entities(self):
        dead_entities: List[AliveEntity] = [
            entity for entity in self.alive_entities_coords if entity.health <= 0 or entity.eaten
        ]
        for entity in dead_entities:
            self._erase_object(obj=entity, where=self._get_object_coordinates(entity))

    @staticmethod
    def _movements_to_coordinates(movement: Movement, from_: Coordinates) -> Coordinates:
        if movement == Movement.STAY:
            return Coordinates(from_.x, from_.y)
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

        return desired_coordinates
