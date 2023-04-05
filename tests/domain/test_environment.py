import itertools

import pytest

from domain.interfaces.entities import AliveEntity, BirthSetup
from evolution.brain import RandomBrain, ControlledBrain
from domain.entities import Herbivore, Predator
from domain.environment import Environment
from domain.service import HerbivoreFoodSustainConstantService
from domain.exceptions import NotVacantPlaceException
from domain.interfaces.setup import HerbivoreFood
from domain.interfaces.objects import Coordinates, Movement, ObservationRange

MOVEMENT_MAPPER_ADJACENT = {
    0: Movement.STAY,
    1: Movement.UP_LEFT,
    2: Movement.UP,
    3: Movement.UP_RIGHT,
    4: Movement.RIGHT,
    5: Movement.DOWN_RIGHT,
    6: Movement.DOWN,
    7: Movement.DOWN_LEFT,
    8: Movement.LEFT,
}

# For movement see MOVEMENT_MAPPER_ADJACENT
MOVEMENT_TEST_CASES = {
    "up_left": {
        "movement": 1,
        "herbivore_coordinates": Coordinates(0, 0),
    },
    "up": {
        "movement": 2,
        "herbivore_coordinates": Coordinates(1, 0),
    },
    "up_right": {
        "movement": 3,
        "herbivore_coordinates": Coordinates(2, 0),
    },
    "left": {
        "movement": 8,
        "herbivore_coordinates": Coordinates(0, 1),
    },
    "stay": {
        "movement": 0,
        "herbivore_coordinates": Coordinates(1, 1),
    },
    "right": {
        "movement": 4,
        "herbivore_coordinates": Coordinates(2, 1),
    },
    "down_left": {
        "movement": 7,
        "herbivore_coordinates": Coordinates(0, 2),
    },
    "down": {
        "movement": 6,
        "herbivore_coordinates": Coordinates(1, 2),
    },
    "down_right": {
        "movement": 5,
        "herbivore_coordinates": Coordinates(2, 2),
    },
}


class TestEnvironment:
    def test_has_space_left_returns_true_when_space_left(self, basic_env):
        basic_env.matrix = [[0, 0, 0], [0, 1, 1], [1, 1, 1]]
        result = basic_env.has_space_left
        assert result is True

    def test_has_space_left_returns_false_when_no_space_left(self, basic_env):
        basic_env.matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = basic_env.has_space_left
        assert result is False

    def test_game_over_no_alive_entity_left(self, basic_herbivore, basic_env):
        basic_env.matrix = [[0, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = basic_env.game_over
        assert result is True

    def test_game_over_one_alive_entity_left(self, basic_herbivore, basic_env):
        basic_env.alive_entities_coords[basic_herbivore] = Coordinates(1, 1)
        result = basic_env.game_over
        assert result is False

    def test_respawn_object_when_place_is_vacant(self, basic_env, basic_herbivore):
        coords = Coordinates(1, 1)
        basic_env._respawn_object(coords, basic_herbivore)
        assert basic_env.matrix[coords.y][coords.x] == basic_herbivore

    def test_respawn_object_when_place_is_not_vacant(self, basic_env, basic_herbivore):
        coords = Coordinates(1, 1)
        basic_env.matrix[coords.y][coords.x] = 1
        with pytest.raises(NotVacantPlaceException):
            basic_env._respawn_object(coords, basic_herbivore)

    def test_create_blank_matrix(self, basic_env):
        matrix = basic_env._create_blank_matrix()
        assert len(matrix) == 16
        assert len(matrix[0]) == 16

        assert matrix[0][0] is None
        assert matrix[0][-1] is None
        assert matrix[-1][0] is None
        assert matrix[-1][-1] is None

        for i in range(1, 15):
            for j in range(1, 15):
                assert matrix[i][j] == 0

    def test_get_object_coordinates_single_object_of_type(self, basic_env, basic_herbivore):
        basic_env.alive_entities_coords[basic_herbivore] = Coordinates(3, 2)
        coords: Coordinates = basic_env._get_object_coordinates(basic_herbivore)
        assert coords.x == 3
        assert coords.y == 2

    def test_get_object_coordinates_multiple_object_of_type(self, basic_env, basic_herbivore):
        basic_env.alive_entities_coords[basic_herbivore] = Coordinates(3, 2)
        basic_env.alive_entities_coords[Herbivore(
            name='Test herbivore 2',
            health=10,
            brain=RandomBrain(),
        )] = Coordinates(0, 0)
        basic_env.alive_entities_coords[Herbivore(
            name='Test herbivore 3',
            health=10,
            brain=RandomBrain(),
        )] = Coordinates(3, 3)
        coords: Coordinates = basic_env._get_object_coordinates(basic_herbivore)
        assert coords.x == 3
        assert coords.y == 2

    def test_erase_object_alive(self, basic_env, basic_herbivore):
        basic_env.matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        basic_env._respawn_object(Coordinates(1, 1), basic_herbivore)
        basic_env._erase_object(basic_herbivore, Coordinates(1, 1))
        assert basic_env.matrix == [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        assert len(basic_env.alive_entities_coords) == 0

    def test_erase_object_not_alive(self, basic_env):
        food = HerbivoreFood(nutrition=3)
        basic_env.matrix = [[1, 1, 1], [1, food, 1], [1, 1, 1]]
        basic_env._erase_object(food, Coordinates(1, 1))
        assert basic_env.matrix == [[1, 1, 1], [1, 0, 1], [1, 1, 1]]

    def test_erase_dead_entities_object_is_eaten(self, basic_env, basic_herbivore):
        basic_env.matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        basic_env._respawn_object(Coordinates(1, 1), basic_herbivore)
        basic_herbivore.was_eaten()
        basic_env._erase_dead_entities()
        assert basic_env.matrix == [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        assert len(basic_env.alive_entities_coords) == 0

    def test_erase_dead_entities_multiple_one_dead_two_alive(self, basic_env):
        basic_env.matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        herb_1 = Herbivore(
            name='Test herbivore 1',
            health=10,
            brain=RandomBrain(),
        )
        herb_2 = Herbivore(
            name='Test herbivore 2',
            health=0,
            brain=RandomBrain(),
        )
        herb_3 = Herbivore(
            name='Test herbivore 3',
            health=5,
            brain=RandomBrain(),
        )

        basic_env._respawn_object(Coordinates(0, 0), herb_1)
        basic_env._respawn_object(Coordinates(1, 1), herb_2)
        basic_env._respawn_object(Coordinates(2, 2), herb_3)

        basic_env._erase_dead_entities()

        assert len(basic_env.alive_entities_coords) == 2
        assert herb_1 in basic_env.alive_entities_coords
        assert herb_2 not in basic_env.alive_entities_coords
        assert herb_3 in basic_env.alive_entities_coords

    def test_erase_dead_entities_multiple_two_dead_on_alive(self, basic_env):
        basic_env.matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        herb_1 = Herbivore(
            name='Test herbivore 1',
            health=10,
            brain=RandomBrain(),
        )
        herb_2 = Herbivore(
            name='Test herbivore 2',
            health=0,
            brain=RandomBrain(),
        )
        herb_3 = Herbivore(
            name='Test herbivore 3',
            health=-1,
            brain=RandomBrain(),
        )

        basic_env._respawn_object(Coordinates(0, 0), herb_1)
        basic_env._respawn_object(Coordinates(1, 1), herb_2)
        basic_env._respawn_object(Coordinates(2, 2), herb_3)

        basic_env._erase_dead_entities()

        assert len(basic_env.alive_entities_coords) == 1
        assert herb_1 in basic_env.alive_entities_coords
        assert herb_2 not in basic_env.alive_entities_coords
        assert herb_3 not in basic_env.alive_entities_coords

    def test_is_empty_coordinates(self, basic_env):
        basic_env.matrix = [[1, 1, 1], [1, 1, 0], [1, 1, 1]]
        assert basic_env._is_empty_coordinates(Coordinates(1, 1)) is False
        assert basic_env._is_empty_coordinates(Coordinates(2, 1)) is True

    def test_get_observation_one_cell_center(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        assert basic_env._get_observation_one_cell_around(Coordinates(2, 2)) == [[7, 8, 9], [12, 13, 14], [17, 18, 19]]

    def test_get_observation_two_cells_center(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        assert basic_env._get_observation_two_cells_around(Coordinates(2, 2)) == [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

    def test_observation_two_cells_corner_upper_left(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        assert basic_env._get_observation_two_cells_around(Coordinates(0, 0)) == [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, 1, 2, 3],
            [None, None, 6, 7, 8],
            [None, None, 11, 12, 13],
        ]

    def test_get_observation_two_cells_corner_down_right(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        assert basic_env._get_observation_two_cells_around(Coordinates(4, 4)) == [
            [13, 14, 15, None, None],
            [18, 19, 20, None, None],
            [23, 24, 25, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ]

    def test_get_living_object_observation_1x_range(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 0, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        herbivore = Herbivore(
            name='Test herb',
            health=10,
            brain=ControlledBrain(observation_width=ObservationRange.ONE_CELL_AROUND),
        )

        basic_env._respawn_object(Coordinates(2, 2), herbivore)

        assert basic_env.get_living_object_observation(herbivore) == [
            [7, 8, 9], [12, herbivore, 14], [17, 18, 19]
        ]

    def test_get_living_object_observation_2x_range(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 0, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        herbivore = Herbivore(
            name='Test herb',
            health=10,
            brain=ControlledBrain(observation_width=ObservationRange.TWO_CELL_AROUND),
        )

        basic_env._respawn_object(Coordinates(2, 2), herbivore)

        assert basic_env.get_living_object_observation(herbivore) == [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, herbivore, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

    def test_set_object_randomly(self, basic_env, basic_herbivore):
        basic_env.matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        basic_env.width = 3
        basic_env.height = 3

        basic_env.set_object_randomly_in_environment(basic_herbivore)
        for row in basic_env.matrix:
            for element in row:
                if element != 0:
                    assert True
                    return
        assert False

    def test_set_obj_near_success_at_corners(self, basic_env, basic_herbivore):
        basic_env.matrix = [[basic_herbivore, 0, 0], [1, 1, 1], [1, 1, 1]]
        basic_env._set_obj_near(Coordinates(0, 0), None)
        assert basic_env.matrix == [[basic_herbivore, 0, None], [1, 1, 1], [1, 1, 1]]

    def test_set_obj_near_success_at_center(self, basic_env, basic_herbivore):
        basic_env.matrix = [[0, 0, 0], [0, basic_herbivore, 0], [0, 0, 0]]
        basic_env._set_obj_near(Coordinates(1, 1), None)
        assert basic_env.matrix == [[None, 0, 0], [0, basic_herbivore, 0], [0, 0, 0]]

    def test_set_obj_near_no_space_around(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, basic_herbivore, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ]
        basic_env.width = 5
        basic_env.height = 5
        basic_env._set_obj_near(Coordinates(1, 1), None)
        assert basic_env.matrix == [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, basic_herbivore, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, None],
        ]

    def test_setup_initial_state(self, basic_env, basic_herbivore, basic_predator):
        basic_env.setup_initial_state([basic_herbivore, basic_predator])
        flatten_matrix = list(itertools.chain(*basic_env.matrix))
        assert len([x for x in flatten_matrix if isinstance(x, AliveEntity)]) == 2
        assert len([x for x in flatten_matrix if isinstance(x, Predator)]) == 1
        assert len([x for x in flatten_matrix if isinstance(x, Herbivore)]) == 1

    def test_setup_initial_state_with_herb_food_sustain_service(self, basic_env, basic_herbivore, basic_predator):
        basic_env.sustain_services = [HerbivoreFoodSustainConstantService(
            required_amount_of_herb_food=10, food_nutrition=3)
        ]
        basic_env.setup_initial_state([basic_herbivore, basic_predator])
        flatten_matrix = list(itertools.chain(*basic_env.matrix))
        assert len([x for x in flatten_matrix if isinstance(x, HerbivoreFood)]) == 10
        assert len([x for x in flatten_matrix if isinstance(x, AliveEntity)]) == 2
        assert len([x for x in flatten_matrix if isinstance(x, Predator)]) == 1
        assert len([x for x in flatten_matrix if isinstance(x, Herbivore)]) == 1

    @pytest.mark.parametrize(
        "test_case",
        MOVEMENT_TEST_CASES.values(),
        ids=MOVEMENT_TEST_CASES.keys(),
    )
    def test_get_next_state_movements(self, basic_env, basic_herbivore, test_case):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        basic_env._respawn_object(Coordinates(1, 1), basic_herbivore)
        basic_herbivore.brain.set_next_movement(test_case['movement'])
        basic_env._get_next_state()
        assert basic_env._get_object_coordinates(basic_herbivore) == test_case['herbivore_coordinates']
        assert (
            basic_env.matrix[1][1] == 0 if test_case['movement'] != 0
            else basic_env.matrix[1][1] == basic_herbivore
        )

    def test_get_next_state_eat_herbivore_food(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        basic_env._respawn_object(Coordinates(1, 0), HerbivoreFood(3))
        basic_env._respawn_object(Coordinates(1, 1), basic_herbivore)
        basic_herbivore.brain.set_next_movement(2)
        basic_env._get_next_state()
        assert basic_env.matrix == [
            [0, basic_herbivore, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        assert basic_herbivore.health == 12

    def test_get_next_state_herbivore_steps_on_predator(self, basic_env, basic_herbivore, basic_predator):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        basic_env._respawn_object(Coordinates(1, 0), basic_predator)
        basic_env._respawn_object(Coordinates(1, 1), basic_herbivore)
        basic_herbivore.brain.set_next_movement(2)
        basic_predator.brain.set_next_movement(0)
        basic_env._get_next_state()
        assert basic_env.matrix == [
            [0, basic_predator, 0],
            [0, basic_herbivore, 0],
            [0, 0, 0]
        ]

    def test_get_next_state_herbivore_steps_on_herbivore(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        another_herbivore = Herbivore(name='Gum', health=10, brain=ControlledBrain())
        basic_env._respawn_object(Coordinates(1, 0), another_herbivore)
        basic_env._respawn_object(Coordinates(1, 1), basic_herbivore)
        another_herbivore.brain.set_next_movement(0)
        basic_herbivore.brain.set_next_movement(2)
        basic_env._get_next_state()
        assert basic_env.matrix == [
            [0, another_herbivore, 0],
            [0, basic_herbivore, 0],
            [0, 0, 0]
        ]

    def test_get_next_state_predator_steps_on_predator(self, basic_env, basic_predator):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        another_predator = Predator(name='Gum', health=10, brain=ControlledBrain())
        basic_env._respawn_object(Coordinates(1, 0), another_predator)
        basic_env._respawn_object(Coordinates(1, 1), basic_predator)
        another_predator.brain.set_next_movement(0)
        basic_predator.brain.set_next_movement(2)
        basic_env._get_next_state()
        assert basic_env.matrix == [
            [0, another_predator, 0],
            [0, basic_predator, 0],
            [0, 0, 0]
        ]

    def test_get_next_state_predator_eat_herbivore(self, basic_env, basic_herbivore, basic_predator):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        basic_env._respawn_object(Coordinates(1, 1), basic_predator)
        basic_env._respawn_object(Coordinates(1, 0), basic_herbivore)
        basic_herbivore.brain.set_next_movement(0)
        basic_predator.brain.set_next_movement(2)
        basic_env._get_next_state()

        assert basic_env.matrix == [
            [0, basic_predator, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        assert len(basic_env.alive_entities_coords) == 1
        assert basic_env not in basic_env.alive_entities_coords
        assert basic_predator.health == 19
        assert basic_herbivore.eaten

    def test_get_next_statee_herbivore_eat_food_and_then_eaten_by_predator(self, basic_env, basic_herbivore, basic_predator):
        basic_env.matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        basic_env._respawn_object(Coordinates(1, 0), basic_herbivore)
        basic_env._respawn_object(Coordinates(1, 1), HerbivoreFood(3))
        basic_env._respawn_object(Coordinates(1, 2), basic_predator)

        basic_herbivore.brain.set_next_movement(6)
        basic_predator.brain.set_next_movement(2)
        basic_env._get_next_state()

        assert basic_env.matrix == [
            [0, 0, 0],
            [0, basic_predator, 0],
            [0, 0, 0]
        ]
        assert len(basic_env.alive_entities_coords) == 1
        assert basic_env not in basic_env.alive_entities_coords
        assert basic_predator.health == 21
        assert basic_herbivore.eaten

    def test_get_next_state_single_herb(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        basic_env._respawn_object(Coordinates(2, 2), basic_herbivore)
        basic_herbivore.brain.set_next_movement(2)  # 2 - up. Look at MOVEMENT_MAPPER_ADJACENT
        assert basic_env._get_next_state() == [
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

    def test_get_next_state_multiple_herb(self, basic_env):
        herb_1 = Herbivore(
            name='Herb 1',
            health=10,
            brain=ControlledBrain(),
        )
        herb_2 = Herbivore(
            name='Herb 2',
            health=10,
            brain=ControlledBrain(),
        )
        herb_3 = Herbivore(
            name='Herb 3',
            health=10,
            brain=ControlledBrain(),
        )
        herb_4 = Herbivore(
            name='Herb 4',
            health=10,
            brain=ControlledBrain(),
        )
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        basic_env._respawn_object(Coordinates(0, 1), herb_1)
        basic_env._respawn_object(Coordinates(3, 0), herb_2)
        basic_env._respawn_object(Coordinates(4, 3), herb_3)
        basic_env._respawn_object(Coordinates(1, 4), herb_4)

        # Look at MOVEMENT_MAPPER_ADJACENT
        herb_1.brain.set_next_movement(2)
        herb_2.brain.set_next_movement(4)
        herb_3.brain.set_next_movement(6)
        herb_4.brain.set_next_movement(8)

        assert basic_env._get_next_state() == [
            [herb_1, 0, 0, 0, herb_2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [herb_4, 0, 0, 0, herb_3],
        ]

    def test_step_living_regime_with_sustain_herb_food_service(self, basic_env, basic_herbivore, basic_predator):
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, HerbivoreFood(3), 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        basic_env.herbivore_food_amount = 1
        basic_env.width, basic_env.height = 5, 5
        basic_env.sustain_services = [HerbivoreFoodSustainConstantService(
            required_amount_of_herb_food=3, food_nutrition=3)
        ]
        basic_env._respawn_object(Coordinates(2, 2), basic_herbivore)
        basic_herbivore.brain.set_next_movement(2)
        basic_env.step_living_regime()
        flatten_matrix = list(itertools.chain(*basic_env.matrix))
        assert len([x for x in flatten_matrix if isinstance(x, HerbivoreFood)]) == 3
        assert basic_env.alive_entities_coords[basic_herbivore] == Coordinates(2, 1)

    def test_get_next_state_herb_give_birth(self):
        basic_env = Environment(window_width=5, window_height=5, sustain_services=[])

        basic_herbivore = Herbivore(
            name='test herb',
            health=100,
            brain=RandomBrain(),
            birth_config=BirthSetup(
                decrease_health_after_birth=10,
                health_after_birth=10,
                birth_after=10,
            ),
        )

        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        basic_env._respawn_object(Coordinates(2, 2), basic_herbivore)
        basic_herbivore.health = 100

        next_state = basic_env._get_next_state()
        flatten_matrix = list(itertools.chain(*next_state))
        assert len([x for x in flatten_matrix if isinstance(x, Herbivore)]) == 2

    def test_step_living_regime_state_entity_died(self, basic_env, basic_herbivore):
        basic_herbivore.health = 1
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        basic_env._respawn_object(Coordinates(2, 2), basic_herbivore)
        basic_herbivore.brain.set_next_movement(2)
        basic_env.step_living_regime()
        assert basic_env.step_living_regime() == ([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], True)

    def test_step_living_regime_game_not_over(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        basic_herbivore.brain.set_next_movement(Movement.UP)
        next_state, finished = basic_env.step_living_regime()

        assert next_state, finished == ([
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], False)

    def test_step_living_regime_game_is_over(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        basic_herbivore.health = 0
        basic_herbivore.brain.set_next_movement(Movement.UP)
        next_state, finished = basic_env.step_living_regime()

        assert next_state, finished == ([
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], True)
