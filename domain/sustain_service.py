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


class HerbivoreSustainService:
    """ Sustain enough amount of herbivores in the env so predator might eat something """

    def initial_sustain(self, environment: Environment) -> None:
        pass

    def subsequent_sustain(self, environment: Environment) -> None:
        pass
