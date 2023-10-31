import os
from dataclasses import dataclass
from functools import partial
from typing import Type, List, Optional

from stable_baselines3 import PPO

from domain.interfaces.environment import SustainEnvironmentService
from evolution.brain import (
    # TrainedBrainPredator100000,
    # TrainedBrainHerbivoreTwoCells1000000,
    # TrainedBrainHerbivoreTwoCells100000,
    BrainForTraining, TrainedModelMixin, get_user_trained_brain,
)
from domain.entities import Predator, Herbivore, EntityType
from domain.environment import Environment
from domain.interfaces.setup import Setup, WindowSetup, EntitySetup, TrainSetup
from domain.interfaces.entities import BirthSetup
from domain.interfaces.objects import Movement, ObservationRange
from domain.service import (
    HerbivoreFoodSustainConstantService,
    HerbivoreFoodSustainEvery3CycleService,
    HerbivoreSustainConstantService,
    TrainedPredatorConstantSustainService
)
from evolution.callbacks import TrainerVisualizer
from evolution.training import HerbivoreTrainer, PredatorTrainer, EntityTrainer
from visualization.visualize import Visualizer


# def get_setup_for_trained_model_predator_and_herb():
#     return Setup(
#         window=WindowSetup(
#             width=50,
#             height=50,
#         ),
#         sustain_services=[
#             # HerbivoreFoodSustainEvery3CycleService(
#             #     initial_food_amount=400, food_nutrition=15,
#             # ),
#             HerbivoreFoodSustainConstantService(required_amount_of_herb_food=400, food_nutrition=10),
#             # HerbivoreSustainConstantService(
#             #     required_amount_of_herbivores=30, initial_herbivore_health=10,
#             # ),
#         ],
#         entities=[
#             EntitySetup(
#                 entity_type=Predator,
#                 entities_amount=50,
#                 initial_health=30,
#                 brain=partial(TrainedBrainPredator100000),
#                 birth=BirthSetup(
#                     decrease_health_after_birth=30,
#                     health_after_birth=10,
#                     birth_after=50,
#                 ),
#             ),
#             EntitySetup(
#                 entity_type=Herbivore,
#                 entities_amount=50,
#                 initial_health=30,
#                 # brain=partial(TrainedBrainHerbivoreTwoCells100000),
#                 # brain=partial(TrainedBrainHerbivoreOneCells100000),
#                 brain=partial(TrainedBrainHerbivoreTwoCells1000000),
#                 birth=BirthSetup(
#                     decrease_health_after_birth=30,
#                     health_after_birth=10,
#                     birth_after=40,
#                 ),
#             ),
#         ]
#     )
#
#
# def get_setup_for_trained_model_herb():
#     return Setup(
#         window=WindowSetup(
#             width=50,
#             height=50,
#         ),
#         sustain_services=[
#             HerbivoreFoodSustainEvery3CycleService(
#                 initial_food_amount=300, food_nutrition=15,
#             )
#         ],
#         entities=[
#             EntitySetup(
#                 entity_type=Herbivore,
#                 entities_amount=5,
#                 brain=partial(TrainedBrainHerbivoreTwoCells100000),
#                 initial_health=10,
#                 birth=BirthSetup(
#                     decrease_health_after_birth=250,
#                     health_after_birth=10,
#                     birth_after=300,
#                 ),
#             ),
#         ]
#     )


# def setup_for_real_time_training_visualization_herb_evolving():
#     window_setup = WindowSetup(
#         width=50, height=50,
#     )
#     basic_herbivore_food_service = HerbivoreFoodSustainConstantService(
#         required_amount_of_herb_food=500, food_nutrition=3,
#     )
#
#     herb_brain = partial(
#         BrainForTraining,
#         train_setup=TrainSetup(
#             learn_frequency=2,
#             learn_timesteps=1000,
#             learn_n_steps=512,
#         ),
#         gym_trainer=HerbivoreTrainer(
#             movement_class=Movement,
#             environment=Environment(
#                 window_width=window_setup.width,
#                 window_height=window_setup.height,
#                 sustain_services=[basic_herbivore_food_service],
#             ),
#             max_live_training_length=3000,
#             health_after_birth=20,
#         )
#     )
#
#     return Setup(
#         window=window_setup,
#         entities=[
#             EntitySetup(
#                 entity_type=Herbivore,
#                 entities_amount=5,
#                 brain=herb_brain,
#                 birth=BirthSetup(
#                     decrease_health_after_birth=10,
#                     health_after_birth=10,
#                     birth_after=15,
#                 ),
#                 initial_health=10,
#             ),
#         ],
#         sustain_services=[HerbivoreFoodSustainEvery3CycleService(food_nutrition=3, initial_food_amount=500)],
#     )
#
#
# def setup_for_real_time_training_visualization_predators_evolving():
#     window_setup = WindowSetup(
#         width=50, height=50,
#     )
#
#     predator_brain = partial(
#         BrainForTraining,
#         train_setup=TrainSetup(
#             learn_frequency=2,
#             learn_timesteps=1000,
#             learn_n_steps=512,
#         ),
#         gym_trainer=PredatorTrainer(
#             movement_class=Movement,
#             environment=Environment(
#                 window_width=window_setup.width,
#                 window_height=window_setup.height,
#                 sustain_services=[
#                     HerbivoreSustainConstantService(required_amount_of_herbivores=100, initial_herbivore_health=10,),
#                 ],
#             ),
#             max_live_training_length=3000,
#             health_after_birth=20,
#         )
#     )
#
#     return Setup(
#         window=window_setup,
#         entities=[
#             EntitySetup(
#                 entity_type=Herbivore,
#                 entities_amount=5,
#                 brain=predator_brain,
#                 birth=BirthSetup(
#                     decrease_health_after_birth=10,
#                     health_after_birth=10,
#                     birth_after=15,
#                 ),
#                 initial_health=10,
#             ),
#         ],
#         sustain_services=[
#             HerbivoreSustainConstantService(required_amount_of_herbivores=100, initial_herbivore_health=10,),
#         ],
#     )


def get_default_sustain_services_factory(
        entity_type: EntityType,
        amount: int,
) -> List[SustainEnvironmentService]:
    if entity_type == EntityType.HERBIVORE:
        return [
            HerbivoreFoodSustainConstantService(required_amount_of_herb_food=amount, food_nutrition=10),
        ]
    elif entity_type == EntityType.PREDATOR:
        return [
            HerbivoreSustainConstantService(required_amount_of_herbivores=amount, initial_herbivore_health=10),
        ]
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def get_default_trainer_factory(
        entity_type: EntityType,
        environment: Environment,
        max_live_training_length: int,
        health_after_birth: int,
        observation_range: ObservationRange,
) -> EntityTrainer:
    if entity_type == EntityType.HERBIVORE:
        return HerbivoreTrainer(
            movement_class=Movement,
            environment=environment,
            max_live_training_length=max_live_training_length,
            health_after_birth=health_after_birth,
            observation_range=observation_range,
        )
    elif entity_type == EntityType.PREDATOR:
        return PredatorTrainer(
            movement_class=Movement,
            environment=environment,
            max_live_training_length=max_live_training_length,
            health_after_birth=health_after_birth,
            observation_range=observation_range,
        )
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def train_the_best_entity(
        window_width: int,
        window_height: int,
        max_live_training_length: int,
        health_after_birth: int,
        total_timestep: int,
        observation_range: ObservationRange,
        entity_type: EntityType,
        save_path: str,
) -> None:
    env = Environment(
        window_width=window_width,
        window_height=window_height,
        sustain_services=get_default_sustain_services_factory(
            entity_type=entity_type,
            amount=int(0.1 * window_width * window_height)
        ),
    )
    gym_trainer: EntityTrainer = get_default_trainer_factory(
        entity_type=entity_type,
        environment=env,
        max_live_training_length=max_live_training_length,
        health_after_birth=health_after_birth,
        observation_range=observation_range,
    )
    model = PPO(
        "MlpPolicy", gym_trainer, verbose=1, tensorboard_log=None,
    )
    model.learn(total_timesteps=total_timestep, progress_bar=True)
    save_path = os.path.join('Training', 'saved_models', save_path)
    model.save(save_path)


def get_setup_for_trained_model_predator_and_herb(
        width: int,
        height: int,
        predator_brain_path: str,
        herbivore_brain_path: Optional[str],
):
    return Setup(
        window=WindowSetup(
            width=50,
            height=50,
        ),
        sustain_services=[
            HerbivoreFoodSustainConstantService(
                required_amount_of_herb_food=int(width * height * 0.1), food_nutrition=10)
        ],
        entities=[
            EntitySetup(
                entity_type=Predator,
                entities_amount=50,
                initial_health=30,
                brain=get_user_trained_brain('path_to_herbivore_brain'),
                birth=BirthSetup(
                    decrease_health_after_birth=30,
                    health_after_birth=10,
                    birth_after=50,
                ),
            ),
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=50,
                initial_health=30,
                brain=get_user_trained_brain('path_to_herbivore_brain'),
                birth=BirthSetup(
                    decrease_health_after_birth=30,
                    health_after_birth=10,
                    birth_after=40,
                ),
            ),
        ]
    )


# @dataclass(frozen=True)
# class EntitySetup:
#     entity_type: Type[AliveEntity]
#     entities_amount: int
#     initial_health: int
#     brain: Callable[[Brain], None]
#     birth: Optional[BirthSetup]

# @dataclass
# class EntitySetup:
#     entity_type: EntityType
#     entities_amount: int
#     initial_health: int
#     brain_path: str
#     birth: BirthSetup


def custom_trained_models_setup(
    window_width: int,
    window_height: int,
    health_after_birth: int,
    total_timestep: int,
    observation_range: ObservationRange,
    entity_type: EntityType,
):
    pass