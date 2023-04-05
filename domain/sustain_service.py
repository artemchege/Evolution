from domain.interfaces.environment import SustainEnvironmentService
from evolution.brain import RandomBrain, TrainedBrainPredator100000
from domain.entities import Herbivore, Predator
from domain.environment import Environment
from domain.interfaces.setup import HerbivoreFood

# TODO: возможно отрефачить, код повторяется из раза в раз, параметризировать фабрику которая бы возвращала састейнеров
# TODO: наверное перенести в service или utils


class HerbivoreFoodSustainEvery3CycleService(SustainEnvironmentService):
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


class HerbivoreFoodSustainEveryCycleService(SustainEnvironmentService):
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


class HerbivoreFoodSustainConstantService(SustainEnvironmentService):
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


class HerbivoreSustainConstantService(SustainEnvironmentService):
    """ Sustain enough amount of herbivores in the env so predator might eat something """

    def __init__(self, required_amount_of_herbivores: int, initial_herbivore_health: int):
        self.required_amount_of_herbivores = required_amount_of_herbivores
        self.initial_herbivore_health = initial_herbivore_health

    def initial_sustain(self, environment: Environment) -> None:
        current_amount: int = environment.herbivores_amount
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


class TrainedPredatorConstantSustainService(SustainEnvironmentService):
    """ Sustain given amount of trained predators in the environment """

    def __init__(self, required_amount_of_predators, initial_predator_health: int):
        self.required_amount_of_predators = required_amount_of_predators
        self.initial_predator_health = initial_predator_health

    def initial_sustain(self, environment: Environment) -> None:
        current_amount: int = environment.predators_amount
        diff_in_amount: int = self.required_amount_of_predators - current_amount

        for _ in range(diff_in_amount):
            environment.set_object_randomly_in_environment(
                Predator(
                    name='Trained predator',
                    health=self.initial_predator_health,
                    brain=TrainedBrainPredator100000(),
                    birth_config=None,
                )
            )

    def subsequent_sustain(self, environment: Environment) -> None:
        self.initial_sustain(environment)
