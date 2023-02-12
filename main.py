from domain.entitites import PrayNoBrain
from domain.environment import Environment, EnvironmentRunner
from visualization.visualize import Visualizer


def main():
    environment = Environment(15, 15)
    man = PrayNoBrain('Artem', 20)
    environment_runner = EnvironmentRunner(environment)
    environment_runner.setup_initial_state(live_objs=[man], pray_foods=50)
    Visualizer(environment_runner).run()


if __name__ == '__main__':
    main()
