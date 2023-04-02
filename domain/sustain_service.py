from domain.brain import RandomBrain
from domain.entitites import Herbivore
from domain.environment import Environment
from domain.objects import HerbivoreFood


class HerbivoreFoodSustainEvery3CycleService:
    """ Watch after HerbivoreFood and replenish """

    def __init__(self, initial_food_amount: int, food_nutrition: int):
        self.initial_food_amount: int = initial_food_amount
        self.food_nutrition = food_nutrition

    def initial_sustain(self, environment: Environment) -> None:
        current_amount: int = environment.herbivore_food_amount
        diff_in_amount: int = self.initial_food_amount - current_amount
        for _ in range(diff_in_amount):
            environment.set_object_randomly_in_environment(HerbivoreFood(self.food_nutrition))
            environment.increment_food_amount()

    def subsequent_sustain(self, environment: Environment):
        if environment.cycle % 3 == 0:
            environment.set_object_randomly_in_environment(HerbivoreFood(self.food_nutrition))
            environment.increment_food_amount()


class HerbivoreFoodSustainEveryCycleService:
    """ Watch after HerbivoreFood and replenish """

    def __init__(self, initial_food_amount: int, food_nutrition: int):
        self.initial_food_amount: int = initial_food_amount
        self.food_nutrition = food_nutrition

    def initial_sustain(self, environment: Environment) -> None:
        current_amount: int = environment.herbivore_food_amount
        diff_in_amount: int = self.initial_food_amount - current_amount
        for _ in range(diff_in_amount):
            environment.set_object_randomly_in_environment(HerbivoreFood(self.food_nutrition))
            environment.increment_food_amount()

    def subsequent_sustain(self, environment: Environment):
        if environment.cycle % 1 == 0:
            environment.set_object_randomly_in_environment(HerbivoreFood(self.food_nutrition))
            environment.increment_food_amount()


class HerbivoreFoodSustainConstantService:
    """ Watch after HerbivoreFood and replenish """

    def __init__(self, required_amount_of_herb_food: int, food_nutrition: int):
        self.required_amount_of_herb_food = required_amount_of_herb_food
        self.food_nutrition = food_nutrition

    def initial_sustain(self, environment: Environment) -> None:
        current_amount: int = environment.herbivore_food_amount
        diff_in_amount: int = self.required_amount_of_herb_food - current_amount
        for _ in range(diff_in_amount):
            environment.set_object_randomly_in_environment(HerbivoreFood(self.food_nutrition))
            environment.increment_food_amount()

    def subsequent_sustain(self, environment: Environment) -> None:
        self.initial_sustain(environment)


class HerbivoreSustainConstantService:
    """ Sustain enough amount of herbivores in the env so predator might eat something """

    def __init__(self, required_amount_of_herbivores: int, initial_herbivore_health: int):
        self.required_amount_of_herbivores = required_amount_of_herbivores
        self.initial_herbivore_health = initial_herbivore_health

    def initial_sustain(self, environment: Environment) -> None:
        current_amount: int = len(
            [herbivore for herbivore in environment.alive_entities_coords.keys() if isinstance(herbivore, Herbivore)]
        )
        diff_in_amount: int = self.required_amount_of_herbivores - current_amount

        for _ in range(diff_in_amount):
            environment.set_object_randomly_in_environment(
                Herbivore(
                    name='Food for predator',
                    health=self.initial_herbivore_health,
                    brain=RandomBrain(),
                    birth_config=None,
                )
            )

    def subsequent_sustain(self, environment: Environment) -> None:
        self.initial_sustain(environment)
