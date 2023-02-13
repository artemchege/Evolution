from domain.entitites import PrayNoBrain
from domain.environment import Environment
from visualization.visualize import Visualizer


def main():
    environment = Environment(15, 15)
    man = PrayNoBrain('Artem', 10)
    environment.setup_initial_state(live_objs=[man], pray_foods=50, nutrition=5)
    Visualizer(environment).run()


if __name__ == '__main__':
    main()
