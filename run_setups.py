import os
from functools import partial
from typing import List, Optional

from stable_baselines3 import PPO

from domain.interfaces.environment import SustainEnvironmentService
from evolution.brain import (
    BrainForTraining, get_user_trained_brain,
)
from domain.entities import Predator, Herbivore, EntityType
from domain.environment import Environment
from domain.interfaces.setup import Setup, WindowSetup, EntitySetup, TrainSetup
from domain.interfaces.entities import BirthSetup
from domain.interfaces.objects import Movement, ObservationRange
from domain.service import (
    HerbivoreFoodSustainConstantService,
    HerbivoreSustainConstantService,
)
from evolution.training import HerbivoreTrainer, PredatorTrainer, EntityTrainer


def setup_for_real_time_training_visualization_herb_evolving(
    width: int,
    height: int,
    amount_of_herb_food: int,
    herb_food_nutrition: int,
    learning_frequency: int,
    learning_timesteps: int,
    learning_n_steps: int,
    health_after_birth: int,
    observation_range: int,
    start_herb_amount: int,
    decrease_parent_health_after_birth: int,
    child_health_after_birth: int,
    birth_after_health_amount: int,
    initial_herb_health: int,
):
    window_setup = WindowSetup(
        width=width, height=height,
    )
    basic_herbivore_food_service = HerbivoreFoodSustainConstantService(
        required_amount_of_herb_food=amount_of_herb_food, food_nutrition=herb_food_nutrition,
    )

    herb_brain = partial(
        BrainForTraining,
        train_setup=TrainSetup(
            learn_frequency=learning_frequency,
            learn_timesteps=learning_timesteps,
            learn_n_steps=learning_n_steps,
        ),
        gym_trainer=HerbivoreTrainer(
            movement_class=Movement,
            environment=Environment(
                window_width=window_setup.width,
                window_height=window_setup.height,
                sustain_services=[basic_herbivore_food_service],
            ),
            max_live_training_length=3000,
            health_after_birth=health_after_birth,
            observation_range=(
                ObservationRange.ONE_CELL_AROUND if observation_range == 1 else ObservationRange.TWO_CELL_AROUND
            ),
        )
    )

    return Setup(
        window=window_setup,
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=start_herb_amount,
                brain=herb_brain,
                birth=BirthSetup(
                    decrease_health_after_birth=decrease_parent_health_after_birth,
                    health_after_birth=child_health_after_birth,
                    birth_after=birth_after_health_amount,
                ),
                initial_health=initial_herb_health,
            ),
        ],
        sustain_services=[basic_herbivore_food_service],
    )


def setup_for_real_time_training_visualization_predator_evolving(
    width: int,
    height: int,
    amount_of_predator_food: int,
    predator_food_nutrition: int,
    learning_frequency: int,
    learning_timesteps: int,
    learning_n_steps: int,
    health_after_birth: int,
    observation_range: int,
    start_pred_amount: int,
    decrease_parent_health_after_birth: int,
    child_health_after_birth: int,
    birth_after_health_amount: int,
    initial_pred_health: int,
) -> Setup:
    window_setup = WindowSetup(
        width=width,
        height=height,
    )

    predator_brain = partial(
        BrainForTraining,
        train_setup=TrainSetup(
            learn_frequency=learning_frequency,
            learn_timesteps=learning_timesteps,
            learn_n_steps=learning_n_steps,
        ),
        gym_trainer=PredatorTrainer(
            movement_class=Movement,
            environment=Environment(
                window_width=window_setup.width,
                window_height=window_setup.height,
                sustain_services=[
                    HerbivoreSustainConstantService(
                        required_amount_of_herbivores=amount_of_predator_food,
                        initial_herbivore_health=predator_food_nutrition,
                    ),
                ],
            ),
            max_live_training_length=3000,
            health_after_birth=health_after_birth,
            observation_range=(
                ObservationRange.ONE_CELL_AROUND if observation_range == 1 else ObservationRange.TWO_CELL_AROUND
            ),
        )
    )

    return Setup(
        window=window_setup,
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=start_pred_amount,
                brain=predator_brain,
                birth=BirthSetup(
                    decrease_health_after_birth=decrease_parent_health_after_birth,
                    health_after_birth=child_health_after_birth,
                    birth_after=birth_after_health_amount,
                ),
                initial_health=initial_pred_health,
            ),
        ],
        sustain_services=[
            HerbivoreSustainConstantService(
                required_amount_of_herbivores=amount_of_predator_food,
                initial_herbivore_health=predator_food_nutrition
            ),
        ],
    )


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
