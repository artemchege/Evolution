import itertools

import pytest

from domain.entitites import AliveEntity, HerbivoreBase, HerbivoreTrain, HerbivoreTrained100000
from domain.environment import NotVacantPlaceException, Environment
from domain.objects import Coordinates, HerbivoreFood, Movement, Setup, WindowSetup, FoodSetup, HerbivoreSetup, \
    HerbivoreTrainSetup

MOVEMENT_TEST_CASES = {
    "up_left": {
        "movement": Movement.UP_LEFT,
        "herbivore_coordinates": Coordinates(0, 0),
    },
    "up": {
        "movement": Movement.UP,
        "herbivore_coordinates": Coordinates(1, 0),
    },
    "up_right": {
        "movement": Movement.UP_RIGHT,
        "herbivore_coordinates": Coordinates(2, 0),
    },
    "left": {
        "movement": Movement.LEFT,
        "herbivore_coordinates": Coordinates(0, 1),
    },
    "stay": {
        "movement": Movement.STAY,
        "herbivore_coordinates": Coordinates(1, 1),
    },
    "right": {
        "movement": Movement.RIGHT,
        "herbivore_coordinates": Coordinates(2, 1),
    },
    "down_left": {
        "movement": Movement.DOWN_LEFT,
        "herbivore_coordinates": Coordinates(0, 2),
    },
    "down": {
        "movement": Movement.DOWN,
        "herbivore_coordinates": Coordinates(1, 2),
    },
    "down_right": {
        "movement": Movement.DOWN_RIGHT,
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
        basic_env.matrix = [[0, basic_herbivore, 1], [1, 1, 1], [1, 1, 1]]
        result = basic_env.game_over
        assert result is False

    def test_game_over_multiple_alive_entity_left(self, basic_herbivore, basic_env):
        basic_env.matrix = [[0, 1, 1], [1, 1, basic_herbivore], [1, 1, basic_herbivore]]
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
        assert len(matrix) == 10
        assert len(matrix[0]) == 10

        assert matrix[0][0] is None
        assert matrix[0][-1] is None
        assert matrix[-1][0] is None
        assert matrix[-1][-1] is None

        for i in range(1, 9):
            for j in range(1, 9):
                assert matrix[i][j] == 0

    def test_get_object_coordinates_single_object_of_type(self, basic_env, basic_herbivore):
        basic_env.matrix[2][3] = basic_herbivore
        coords: Coordinates = basic_env._get_object_coordinates(basic_herbivore)
        assert coords.x == 3
        assert coords.y == 2

    def test_get_object_coordinates_multiple_object_of_type(self, basic_env, basic_herbivore):
        basic_env.matrix[2][3] = basic_herbivore
        basic_env.matrix[0][0] = HerbivoreBase(
            name='Test herbivore 2',
            health=10,
        )
        basic_env.matrix[3][3] = HerbivoreBase(
            name='Test herbivore 3',
            health=10,
        )
        coords: Coordinates = basic_env._get_object_coordinates(basic_herbivore)
        assert coords.x == 3
        assert coords.y == 2

    def test_erase_object(self, basic_env, basic_herbivore):
        basic_env.matrix = [[1, 1, 1], [1, 1, basic_herbivore], [1, 1, 1]]
        basic_env._erase_object(Coordinates(2, 1))
        assert basic_env.matrix == [[1, 1, 1], [1, 1, 0], [1, 1, 1]]

    def test_is_empty_coordinates(self, basic_env):
        basic_env.matrix = [[1, 1, 1], [1, 1, 0], [1, 1, 1]]
        assert basic_env._is_empty_coordinates(Coordinates(1, 1)) is False
        assert basic_env._is_empty_coordinates(Coordinates(2, 1)) is True

    def test_get_observation(self, basic_env):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        assert basic_env._get_observation(Coordinates(2, 2)) == [[7, 8, 9], [12, 13, 14], [17, 18, 19]]

    def test_get_living_object_observation(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, basic_herbivore, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]

        assert basic_env.get_living_object_observation(basic_herbivore) == [
            [7, 8, 9], [12, basic_herbivore, 14], [17, 18, 19]
        ]

    def test_set_object_randomly(self, basic_env, basic_herbivore):
        basic_env.matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        basic_env.width = 3
        basic_env.height = 3

        basic_env._set_object_randomly(basic_herbivore)
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

    def test_setup_initial_state(self, basic_env, basic_herbivore):
        basic_env.setup_initial_state(herbivores=[basic_herbivore])
        flatten_matrix = list(itertools.chain(*basic_env.matrix))
        assert len([x for x in flatten_matrix if isinstance(x, HerbivoreFood)]) == 3
        assert len([x for x in flatten_matrix if isinstance(x, AliveEntity)]) == 1

    @pytest.mark.parametrize(
        "test_case",
        MOVEMENT_TEST_CASES.values(),
        ids=MOVEMENT_TEST_CASES.keys(),
    )
    def test_make_move_up_left(self, basic_env, basic_herbivore, test_case):
        basic_env.matrix = [
            [0, 0, 0],
            [0, basic_herbivore, 0],
            [0, 0, 0]
        ]
        basic_env._make_move(test_case['movement'], basic_herbivore)
        assert basic_env._get_object_coordinates(basic_herbivore) == test_case['herbivore_coordinates']
        assert (
            basic_env.matrix[1][1] == 0 if test_case['movement'] != Movement.STAY
            else basic_env.matrix[1][1] == basic_herbivore
        )

    def test_make_move_eat_herbivore_food(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, HerbivoreFood(3), 0],
            [0, basic_herbivore, 0],
            [0, 0, 0]
        ]
        basic_env._make_move(Movement.UP, basic_herbivore)
        assert basic_env.matrix == [
            [0, basic_herbivore, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        assert basic_herbivore.health == 13

    def test_get_next_state_single_herb(self, basic_env, basic_herbivore):
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        basic_herbivore.brain.set_next_movement(Movement.UP)

        assert basic_env._get_next_state() == [
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

    def test_get_next_state_multiple_herb(self, basic_env):
        herb_1 = HerbivoreBase(
            name='Herb 1',
            health=10,
        )
        herb_2 = HerbivoreBase(
            name='Herb 2',
            health=10,
        )
        herb_3 = HerbivoreBase(
            name='Herb 3',
            health=10,
        )
        herb_4 = HerbivoreBase(
            name='Herb 4',
            health=10,
        )
        basic_env.matrix = [
            [0, 0, 0, herb_2, 0],
            [herb_1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, herb_3],
            [0, herb_4, 0, 0, 0],
        ]
        herb_1.brain.set_next_movement(Movement.UP)
        herb_2.brain.set_next_movement(Movement.RIGHT)
        herb_3.brain.set_next_movement(Movement.DOWN)
        herb_4.brain.set_next_movement(Movement.LEFT)

        assert basic_env._get_next_state() == [
            [herb_1, 0, 0, 0, herb_2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [herb_4, 0, 0, 0, herb_3],
        ]

    def test_get_next_state_entity_died(self, basic_env, basic_herbivore):
        basic_herbivore.health = 0
        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        basic_herbivore.brain.set_next_movement(Movement.UP)

        assert basic_env._get_next_state() == [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

    def test_get_next_state_herb_give_birth(self):
        basic_env = Environment(
            setup=Setup(
                window=WindowSetup(
                    width=5,
                    height=5,
                ),
                food=FoodSetup(
                    herbivore_food_amount=3,
                    herbivore_food_nutrition=3,
                    replenish_food=False,
                ),
                herbivore=HerbivoreSetup(
                    herbivores_amount=1,
                    herbivore_class=HerbivoreTrain,
                    herbivore_initial_health=20,
                    birth_after=10,
                    learn_frequency=2,
                    learn_n_steps=512,
                ),
                train=HerbivoreTrainSetup(
                    herbivore_trainer_class=HerbivoreBase,
                    max_live_training_length=5000,
                )
            )
        )

        basic_herbivore = HerbivoreTrain(
            name='test herb',
            health=100,
            environment=basic_env,
        )

        basic_env.matrix = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, basic_herbivore, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        basic_herbivore.health = 100

        next_state = basic_env._get_next_state()
        flatten_matrix = list(itertools.chain(*next_state))
        assert len([x for x in flatten_matrix if isinstance(x, HerbivoreBase)]) == 2

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
