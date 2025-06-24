"""Botón interactivo para la interfaz"""

import pygame

from .button_config import ButtonConfig


class InteractiveButton:
    """Botón interactivo"""

    def __init__(self, config: ButtonConfig):
        self.config = config
        self.rect = pygame.Rect(config.x, config.y, config.width, config.height)
        self.is_hovered = False
        self.is_pressed = False

        # Configurar fuente
        try:
            self.font = pygame.font.SysFont("arial", config.font_size)
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, config.font_size)

    def handle_event(self, event):
        """Maneja eventos del botón"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_pressed:
                self.is_pressed = False
                if self.rect.collidepoint(event.pos):
                    self.config.action()
                    return True
        return False

    def draw(self, surface):
        """Dibuja el botón con colores estáticos y gradientes"""
        # Determinar color según estado
        if self.is_pressed:
            color = self.config.color_active
        elif self.is_hovered:
            color = self.config.color_hover
        else:
            color = self.config.color_normal

        # Dibujar fondo del botón con bordes redondeados
        pygame.draw.rect(surface, color, self.rect, border_radius=10)

        # Borde simple
        border_color = (
            self.config.color_normal if not self.is_hovered else self.config.color_hover
        )
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=10)

        # Dibujar texto centrado
        text_surface = self.font.render(self.config.text, True, self.config.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
