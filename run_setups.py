from functools import partial

from stable_baselines3 import PPO

from domain.brain import TrainedBrainPredator100000, TrainedBrainHerbivoreTwoCells1000000, \
    TrainedBrainHerbivoreTwoCells100000
from domain.entitites import Predator, Herbivore
from domain.environment import Environment
from domain.objects import Setup, WindowSetup, EntitySetup, BirthSetup, TrainSetup, Movement
from domain.sustain_service import HerbivoreFoodSustainConstantService, HerbivoreFoodSustainEvery3CycleService, \
    HerbivoreSustainConstantService, TrainedPredatorConstantSustainService
from evolution.callbacks import TrainerVisualizer
from evolution.training import BrainForTraining, HerbivoreTrainer, PredatorTrainer


def get_setup_for_trained_model_predator_and_herb():
    return Setup(
        window=WindowSetup(
            width=50,
            height=50,
        ),
        sustain_services=[
            # HerbivoreFoodSustainEvery3CycleService(
            #     initial_food_amount=400, food_nutrition=15,
            # ),
            HerbivoreFoodSustainConstantService(required_amount_of_herb_food=400, food_nutrition=10),
            # HerbivoreSustainConstantService(
            #     required_amount_of_herbivores=30, initial_herbivore_health=10,
            # ),
        ],
        entities=[
            EntitySetup(
                entity_type=Predator,
                entities_amount=50,
                initial_health=30,
                brain=partial(TrainedBrainPredator100000),
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
                # brain=partial(TrainedBrainHerbivoreTwoCells100000),
                # brain=partial(TrainedBrainHerbivoreOneCells100000),
                brain=partial(TrainedBrainHerbivoreTwoCells1000000),
                birth=BirthSetup(
                    decrease_health_after_birth=30,
                    health_after_birth=10,
                    birth_after=40,
                ),
            ),
        ]
    )


def get_setup_for_trained_model_herb():
    return Setup(
        window=WindowSetup(
            width=50,
            height=50,
        ),
        sustain_services=[
            HerbivoreFoodSustainEvery3CycleService(
                initial_food_amount=300, food_nutrition=15,
            )
        ],
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=5,
                brain=partial(TrainedBrainHerbivoreTwoCells100000),
                initial_health=10,
                birth=BirthSetup(
                    decrease_health_after_birth=250,
                    health_after_birth=10,
                    birth_after=300,
                ),
            ),
        ]
    )


def setup_for_real_time_training_visualization_herb_evolving():
    window_setup = WindowSetup(
        width=50, height=50,
    )
    basic_herbivore_food_service = HerbivoreFoodSustainConstantService(
        required_amount_of_herb_food=500, food_nutrition=3,
    )

    herb_brain = partial(
        BrainForTraining,
        train_setup=TrainSetup(
            learn_frequency=2,
            learn_timesteps=1000,
            learn_n_steps=512,
        ),
        gym_trainer=HerbivoreTrainer(
            movement_class=Movement,
            environment=Environment(
                window_width=window_setup.width,
                window_height=window_setup.height,
                sustain_services=[basic_herbivore_food_service],
            ),
            max_live_training_length=3000,
            health_after_birth=20,
        )
    )

    return Setup(
        window=window_setup,
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=5,
                brain=herb_brain,
                birth=BirthSetup(
                    decrease_health_after_birth=10,
                    health_after_birth=10,
                    birth_after=15,
                ),
                initial_health=10,
            ),
        ],
        sustain_services=[HerbivoreFoodSustainEvery3CycleService(food_nutrition=3, initial_food_amount=500)],
    )


def setup_for_real_time_training_visualization_predators_evolving():
    window_setup = WindowSetup(
        width=50, height=50,
    )

    predator_brain = partial(
        BrainForTraining,
        train_setup=TrainSetup(
            learn_frequency=2,
            learn_timesteps=1000,
            learn_n_steps=512,
        ),
        gym_trainer=PredatorTrainer(
            movement_class=Movement,
            environment=Environment(
                window_width=window_setup.width,
                window_height=window_setup.height,
                sustain_services=[
                    HerbivoreSustainConstantService(required_amount_of_herbivores=100, initial_herbivore_health=10,),
                ],
            ),
            max_live_training_length=3000,
            health_after_birth=20,
        )
    )

    return Setup(
        window=window_setup,
        entities=[
            EntitySetup(
                entity_type=Herbivore,
                entities_amount=5,
                brain=predator_brain,
                birth=BirthSetup(
                    decrease_health_after_birth=10,
                    health_after_birth=10,
                    birth_after=15,
                ),
                initial_health=10,
            ),
        ],
        sustain_services=[
            HerbivoreSustainConstantService(required_amount_of_herbivores=100, initial_herbivore_health=10,),
        ],
    )


def train_best_herbivore():
    env = Environment(
            window_width=20,
            window_height=20,
            sustain_services=[
                HerbivoreFoodSustainEvery3CycleService(
                    initial_food_amount=60, food_nutrition=3,
                ),
                TrainedPredatorConstantSustainService(
                    required_amount_of_predators=5, initial_predator_health=100,
                )
            ],
    )

    gym_trainer = HerbivoreTrainer(
        movement_class=Movement,
        environment=env,
        max_live_training_length=1000,
        health_after_birth=20,
        # visualizer=Visualizer(env),
    )

    # dummy_trainer = DummyVecEnv([lambda: gym_trainer])

    model = PPO(
        "MlpPolicy", gym_trainer, verbose=1, tensorboard_log=None,
    )
    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=TrainerVisualizer())  # TrainerVisualizator()
    # save_path = os.path.join('Training', 'saved_models', 'PPO_model_Herbivore_100000)_20x20_food60_3_two_cells')
    # model.save(save_path)


def train_best_predator():
    gym_trainer = PredatorTrainer(
        movement_class=Movement,
        environment=Environment(
            window_width=20,
            window_height=20,
            sustain_services=[HerbivoreSustainConstantService(
                    required_amount_of_herbivores=30, initial_herbivore_health=10,
                )
            ],
        ),
        max_live_training_length=3000,
        health_after_birth=20,
    )
    model = PPO(
        "MlpPolicy", gym_trainer, verbose=1, tensorboard_log=None,
    )
    model.learn(total_timesteps=100000, progress_bar=True)
    # save_path = os.path.join('Training', 'saved_models', 'PPO_model_Predator_100000_20x20_food30')
    # model.save(save_path)