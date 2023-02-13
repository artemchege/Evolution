from typing import List

import pygame

from contrib.utils import logger
from domain.entitites import AliveEntity
from domain.environment import Environment
from domain.objects import Coordinates, PrayFood
from visualization.constants import GREY_DARK, GREY_LIGHT, GREEN, BLUE, BLACK


class Visualizer:
    def __init__(self, runner: Environment):
        pygame.display.set_caption('AI')

        self.window_width = 800
        self.window_height = 800
        self.FPS = 1
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.environment_runner: Environment = runner

        self.number_of_rows: int = self.environment_runner.height
        self.number_of_columns: int = self.environment_runner.width
        self.cell_width: int = self.window_width // self.number_of_columns
        self.cell_height: int = self.window_height // self.number_of_rows

    def create_blank_space(self):
        self.window.fill(GREY_DARK)
        for row in range(self.number_of_rows):
            for col in range(self.number_of_columns):
                pygame.draw.rect(
                    self.window,
                    GREY_LIGHT,
                    (row * self.cell_width, col * self.cell_height, self.cell_width, self.cell_height),
                    border_radius=20,
                )

    def render(self, matrix: List[List]):
        for y, row in enumerate(matrix):
            for x, element in enumerate(row):
                cell_center = Coordinates(
                    y * self.cell_height + self.cell_height // 2,
                    x * self.cell_width + self.cell_width // 2,
                )
                radius: int = self.cell_width // 2

                if isinstance(element, AliveEntity):
                    pygame.draw.circle(self.window, BLUE, (cell_center.x, cell_center.y), radius)

                if isinstance(element, PrayFood):
                    pygame.draw.circle(self.window, GREEN, (cell_center.x, cell_center.y), radius)

                if element is None:
                    pygame.draw.circle(self.window, BLACK, (cell_center.x, cell_center.y), radius)

    def run(self):
        run = True
        clock = pygame.time.Clock()

        while run:
            clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    logger.debug(f'Touches are not supported')

            state_to_render, _ = self.environment_runner.step_forward()
            self.create_blank_space()
            self.render(state_to_render)
            pygame.display.update()

        pygame.quit()
        logger.debug('Game was closed')

