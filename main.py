from domain.entitites import PrayNoBrain
from domain.environment import Environment
from visualization.visualize import Visualizer


def main():
    environment = Environment(15, 15)
    pray_1 = PrayNoBrain('Mammoth', 10)
    pray_2 = PrayNoBrain('Dodo bird', 15)
    environment.setup_initial_state(live_objs=[pray_1, pray_2], pray_foods=50, nutrition=5)
    Visualizer(environment).run()


if __name__ == '__main__':
    main()
