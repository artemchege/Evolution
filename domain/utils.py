import json
from typing import List

from domain.interfaces.environment import EnvironmentInterface


class StatisticsCollector:

    def __init__(self, environment: EnvironmentInterface, filename: str):
        self.environment: EnvironmentInterface = environment
        self.filename: str = filename
        self.snapshots: List[dict] = []

    def make_snapshot(self):
        self.snapshots.append(
            {
                "cycle": self.environment.cycle,
                "alive_entities": len(self.environment.alive_entities_coords),
                "herbivores_amount": self.environment.herbivores_amount,
                "predators_amount": self.environment.predators_amount,
                "herbivore_food": self.environment.herbivore_food_amount,
            }
        )

    def dump_to_file(self):
        with open(f"statistics/{self.filename}.json", "w") as file:
            json.dump(self.snapshots, file)
