from typing import List

import pygame

from contrib.utils import logger
from domain.entitites import AliveEntity
from domain.environment import Environment
from domain.objects import Coordinates, HerbivoreFood
from visualization.constants import GREY_DARK, GREY_LIGHT, GREEN, BLUE, BLACK


class Visualizer:
    def __init__(self, env: Environment):
        pygame.display.set_caption('AI')

        self.window_width = 800
        self.window_height = 800
        self.FPS = 30
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.env: Environment = env

        self.number_of_rows: int = self.env.height
        self.number_of_columns: int = self.env.width
        self.cell_width: int = self.window_width // self.number_of_columns
        self.cell_height: int = self.window_height // self.number_of_rows

        self.clock = pygame.time.Clock()

    def render_step(self, current_global_env_state: List[List]):
        """ Main interface, accepts environment matrix without any transformation, creates visualization """

        self._check_keyboard_events()
        self._create_blank_space()
        self._render(current_global_env_state)
        pygame.display.update()
        self.clock.tick(self.FPS)

    def run(self):
        """ For standalone usage without stable baseline3 and Gym """

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            state_to_render, _ = self.env.step_living_regime()
            self.render_step(state_to_render)

        pygame.quit()
        logger.debug('Game was closed')

    def _create_blank_space(self):
        self.window.fill(GREY_DARK)
        for row in range(self.number_of_rows):
            for col in range(self.number_of_columns):
                pygame.draw.rect(
                    self.window,
                    GREY_LIGHT,
                    (row * self.cell_width, col * self.cell_height, self.cell_width, self.cell_height),
                    border_radius=20,
                )

    def _render(self, matrix: List[List]):
        for y, row in enumerate(matrix):
            for x, element in enumerate(row):
                cell_center = Coordinates(
                    y * self.cell_height + self.cell_height // 2,
                    x * self.cell_width + self.cell_width // 2,
                )
                radius: int = self.cell_width // 2

                if isinstance(element, AliveEntity):
                    pygame.draw.circle(self.window, BLUE, (cell_center.x, cell_center.y), radius)

                if isinstance(element, HerbivoreFood):
                    pygame.draw.circle(self.window, GREEN, (cell_center.x, cell_center.y), radius)

                if element is None:
                    pygame.draw.circle(self.window, BLACK, (cell_center.x, cell_center.y), radius)

    def _check_keyboard_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.FPS = 1  # set slow rendering
                elif event.key == pygame.K_f:
                    self.FPS = 30

