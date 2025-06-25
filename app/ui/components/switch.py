"""Switch/toggle interactivo para la interfaz"""

import pygame

from .button_config import ButtonConfig


class InteractiveSwitch:
    """Switch/toggle interactivo"""

    def __init__(self, config: ButtonConfig):
        self.config = config
        self.rect = pygame.Rect(config.x, config.y, config.width, config.height)
        self.is_hovered = False
        self.is_pressed = False
        self.state = config.switch_state

        # Configurar fuente
        try:
            self.font = pygame.font.SysFont("arial", config.font_size)
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, config.font_size)

    def handle_event(self, event):
        """Maneja eventos del switch - solo en el área del track"""
        # Definir el área clickeable del track (solo el switch)
        track_width = 50
        track_height = 20
        track_x = self.rect.right - track_width - 5
        track_y = self.rect.centery - track_height // 2
        track_rect = pygame.Rect(track_x, track_y, track_width, track_height)

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = track_rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and track_rect.collidepoint(event.pos):
                self.is_pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_pressed:
                self.is_pressed = False
                if track_rect.collidepoint(event.pos):
                    self.state = not self.state
                    self.config.action()
                    return True
        return False

    def draw(self, surface):
        """Dibuja el switch con formato: 'Texto: [Switch]'"""
        # Track del switch (carril donde se mueve el indicador)
        track_width = 50
        track_height = 20
        track_x = self.rect.right - track_width - 5  # 5px margen derecho
        track_y = self.rect.centery - track_height // 2  # Centrado verticalmente

        # Crear rectángulo del track
        track_rect = pygame.Rect(track_x, track_y, track_width, track_height)

        # Color del track según estado
        track_color = self.config.color_switch_on if self.state else (80, 80, 80)
        pygame.draw.rect(surface, track_color, track_rect, border_radius=10)

        # Indicador móvil (círculo que se desliza)
        indicator_radius = 8
        if self.state:  # Activado, el indicador se mueve a la derecha
            indicator_x = track_rect.right - indicator_radius - 2
            indicator_color = (255, 255, 220)
            indicator_border = (255, 200, 100)
        else:  # Desactivado, el indicador se mueve a la izquierda
            indicator_x = track_rect.left + indicator_radius + 2
            indicator_color = (200, 200, 200)
            indicator_border = (160, 160, 160)

        indicator_y = track_rect.centery

        # Dibujar indicador principal
        pygame.draw.circle(
            surface, indicator_color, (indicator_x, indicator_y), indicator_radius
        )
        pygame.draw.circle(
            surface, indicator_border, (indicator_x, indicator_y), indicator_radius, 2
        )

        # Texto del switch
        if self.state:
            text_color = (220, 255, 200)  # Verde claro cuando ON
        else:
            text_color = (200, 200, 200)  # Gris claro cuando OFF

        # Formatear texto como "Texto:"
        display_text = f"{self.config.text}:"
        text_surface = self.font.render(display_text, True, text_color)

        text_rect = text_surface.get_rect()
        text_rect.centery = self.rect.centery
        text_rect.x = self.rect.x + 5  # Pequeño margen izquierdo
        surface.blit(text_surface, text_rect)
