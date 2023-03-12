import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from contrib.utils import logger
from domain.entitites import HerbivoreBase, HerbivoreTrained100000, HerbivoreTrain
from domain.environment import EnvironmentTrainRegime, EnvironmentLiveRegime
from domain.objects import Movement, Coordinates, MOVEMENT_MAPPER_ADJACENT, TrainingSetup
from evolution.training import HerbivoreGym
from visualization.visualize import Visualizer


def visualize_trained_model_visualizer_runner():
    # Визуализация тренированной модели через мое окружение
    environment = EnvironmentLiveRegime(
        width=15,
        height=15,
        replenish_food=True,
        herbivore_food_amount=50,
        food_nutrition=3,
    )
    # pray_1 = HerbivoreBase('Mammoth', 10)
    # pray_2 = HerbivoreBase('Dodo bird', 15)
    # trained_prey = HerbivoreTrained100000('Crow', 10)
    herb_tr_1 = HerbivoreTrain(
        name='Lolka',
        health=20,
        env=environment,
    )
    # herb_tr_2 = HerbivoreTrain(
    #     name='Kelka',
    #     health=10,
    #     env=environment,
    # )
    environment.setup_initial_state(live_objs=[herb_tr_1])   # todo: возможно отвественность раннера, который надо создать
    Visualizer(environment).run()


def visualize_training():
    # Визуализация тренировки
    setup: TrainingSetup = TrainingSetup(
        herbivore_food_amount=50,
        herbivore_food_nutrition=2,
        replenish_food=True,
        living_object_name='Mammoth',
        living_object_class=HerbivoreBase,
        living_object_initial_health=10,
        live_length=5000,
    )
    episodes = 5000
    train_env_v = HerbivoreGym(
        movement_class=Movement,
        environment=EnvironmentTrainRegime(
            width=16,
            height=16,
            replenish_food=setup.replenish_food,
            food_nutrition=setup.herbivore_food_nutrition,
        ),
        setup=setup,
    )
    model = PPO("MlpPolicy", train_env_v, verbose=1, tensorboard_log=None)
    for i in range(episodes):
        print(f'Episode {i} started')
        observ = train_env_v.reset()
        done = False
        score = 0

        # Подменяем окружения и обучаем модель, как закончили, меняем окружение обратно и визуализируем, получается
        # так что на отдельном окружении тренируем, на отдельном показываем, как только в режиме показа сущность
        # умерла - отправляем на новую тренировку. _v - визуализация, _l - обучение.
        train_env_l = HerbivoreGym(
            movement_class=Movement,
            environment=EnvironmentTrainRegime(
                width=16,
                height=16,
                replenish_food=True,
                food_nutrition=setup.herbivore_food_nutrition,
            ),
            setup=setup,
        )
        model.set_env(train_env_l)
        model.learn(total_timesteps=1)
        model.set_env(train_env_v)

        while not done:
            action, _ = model.predict(observ)
            observ, reward, done, info = train_env_v.step(int(action))
            score += reward
            train_env_v.render()

        print('Episode {} Finished. Score:{}'.format(i, score))
        train_env_v.close()


def visualize_trained_model_evaluate_policy():
    # Визуализация оттренированной модели через evaluate_policy
    save_path = os.path.join('Training', 'saved_models', 'PPO_model_Pray_100000')
    model = PPO.load(save_path)
    # environment = Environment(16, 16)
    # train_env = HerbivoreTrainingRunner(Movement, environment)
    # evaluate_policy(model, train_env, n_eval_episodes=10, render=True)


def benchmark_herbivore_no_brain():
    # Бенчмарк травоядного без мозга
    results_lived_for = []
    for episode in range(500):
        pass

        # setup
        # environment = Environment(15, 15)
        # pray = PrayNoBrain('Mammoth', 10)
        # environment.setup_initial_state(live_objs=[pray], pray_foods=50, nutrition=5)
        #
        # # run
        # game_over = False
        # while not game_over:
        #     _, game_over = environment.step_living_regime()
        # results_lived_for.append(pray.lived_for)

    logger.info(f'Average lived for is: {sum(results_lived_for)/len(results_lived_for)}')


def main():
    visualize_trained_model_visualizer_runner()


if __name__ == '__main__':
    main()
