from typing import List

import pygame

from domain.entitites import AliveEntity
from domain.environment import Environment
from domain.objects import Coordinates, HerbivoreFood
from visualization.constants import GREY_DARK, GREY_LIGHT, GREEN, BLUE, BLACK


class Visualizer:
    def __init__(self, env: Environment):
        pygame.display.set_caption('AI')
        pygame.font.init()

        self.field_width = 800
        self.statistic_block_height = 100
        self.field_height = 800
        self.FPS = 1
        self.window = pygame.display.set_mode((self.field_width, self.field_height + self.statistic_block_height))
        self.env: Environment = env
        self.large_font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 12)

        self.number_of_rows: int = self.env.height
        self.number_of_columns: int = self.env.width
        self.cell_width: int = self.field_width // self.number_of_columns
        self.cell_height: int = self.field_height // self.number_of_rows

        self.clock = pygame.time.Clock()

    def render_step(self, current_global_env_state: List[List]):
        """ Main interface, accepts environment matrix without any transformation, creates visualization """

        self._check_keyboard_events()
        self._create_blank_space()
        self._render(current_global_env_state)
        self._render_stat()
        pygame.display.update()
        self.clock.tick(self.FPS)

    def _render_stat(self):
        statistics_rect = pygame.draw.rect(
            self.window,
            GREY_LIGHT,
            (0, self.field_width, self.field_width, self.statistic_block_height),
            border_radius=20,
        )

        text_to_render = f"Current cycle: {self.env.cycle}. " \
                         f"Herbivores: {len(self.env.alive_entities_coords)}. " \
                         f"Food: {self.env.herbivore_food_amount}."
        text = self.large_font.render(text_to_render, True, BLACK)
        text_rect = text.get_rect(center=statistics_rect.center)
        self.window.blit(text, text_rect)

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
                    entity_circle = pygame.draw.circle(self.window, BLUE, (cell_center.x, cell_center.y), radius)
                    text = self.small_font.render(str(element.health), True, BLACK)
                    text_rect = text.get_rect(center=entity_circle.center)
                    self.window.blit(text, text_rect)

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

