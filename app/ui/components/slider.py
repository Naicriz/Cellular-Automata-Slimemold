"""Slider interactivo modernizado para valores numéricos"""

import pygame

from app.ui.components.button_config import ButtonConfig


class InteractiveSlider:
    """Slider interactivo con diseño modernizado"""

    def __init__(
        self,
        x,
        y,
        width,
        height,
        min_val,
        max_val,
        current_val,
        label,
        callback,
        description="",
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.description = description
        self.callback = callback
        self.is_dragging = False
        self.is_hovered = False

        # Handle del slider
        self.handle_radius = 10

        # Crear configuración de colores
        self.config = ButtonConfig(0, 0, 0, 0, "", None)

        # Configurar fuentes
        try:
            self.font = pygame.font.SysFont("arial", 14)
            self.desc_font = pygame.font.SysFont("arial", 16, bold=True)
            self.label_font = pygame.font.Font(None, 24)
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, 16)
            self.desc_font = pygame.font.Font(None, 18)
            self.label_font = pygame.font.Font(None, 24)

    def handle_event(self, event):
        """Maneja eventos del slider"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            if self.is_dragging:
                # Calcular nuevo valor basado en posición del mouse
                mouse_x = event.pos[0]
                relative_x = mouse_x - self.rect.x
                relative_x = max(0, min(self.rect.width, relative_x))

                # Convertir posición a valor
                ratio = relative_x / self.rect.width
                new_val = self.min_val + (self.max_val - self.min_val) * ratio

                if isinstance(self.min_val, int):
                    new_val = int(new_val)
                else:
                    new_val = round(new_val, 1)

                if new_val != self.current_val:
                    self.current_val = new_val
                    self.callback(new_val)
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.is_dragging = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.is_dragging = False

        return False

    def draw(self, surface):
        """Dibuja el slider con diseño modernizado"""
        # Texto descriptivo arriba del slider
        if self.description:
            desc_text = self.desc_font.render(
                self.description, True, self.config.slider_label_color
            )
            desc_rect = desc_text.get_rect()
            desc_rect.centerx = self.rect.centerx
            desc_rect.bottom = self.rect.top - 22
            surface.blit(desc_text, desc_rect)

        # Label con valor actual arriba del track
        label_text = f"{self.label}: {int(self.current_val)}"
        label_surface = self.label_font.render(label_text, True, self.config.text_color)
        label_rect = label_surface.get_rect()
        label_rect.x = self.rect.x
        label_rect.y = self.rect.y - 25
        surface.blit(label_surface, label_rect)

        # Track del slider (línea principal)
        track_color = (
            self.config.slider_track_hover
            if self.is_hovered
            else self.config.slider_track_color
        )

        # Línea de track principal
        pygame.draw.line(
            surface,
            track_color,
            (self.rect.left, self.rect.centery),
            (self.rect.right, self.rect.centery),
            4,
        )

        # Calcular posición del handle
        ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.left + int(ratio * self.rect.width)

        # Línea de progreso (desde el inicio hasta el handle)
        if ratio > 0:
            pygame.draw.line(
                surface,
                self.config.slider_progress_color,
                (self.rect.left, self.rect.centery),
                (handle_x, self.rect.centery),
                4,
            )

        # Handle del slider (círculo)
        handle_color = (
            self.config.slider_handle_active
            if self.is_dragging
            else self.config.slider_handle_color
        )

        # Dibujar handle principal
        pygame.draw.circle(
            surface, handle_color, (handle_x, self.rect.centery), self.handle_radius
        )

        # Borde del handle
        pygame.draw.circle(
            surface,
            self.config.slider_handle_border,
            (handle_x, self.rect.centery),
            self.handle_radius,
            2,
        )


# Clase alternativa más simple para compatibilidad
class Slider:
    """Slider simple con paleta modernizada"""

    def __init__(self, x, y, w, min_val, max_val, initial, label):
        self.rect = pygame.Rect(x, y, w, 20)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.handle_radius = 10
        self.dragging = False
        self.hovered = False
        self.font = pygame.font.Font(None, 24)

        # Configuración de colores modernizada
        self.config = ButtonConfig(0, 0, 0, 0, "", None)

    def draw(self, surface):
        """Dibuja el slider con colores modernizados"""
        # Track del slider
        track_color = (
            self.config.slider_track_hover
            if self.hovered
            else self.config.slider_track_color
        )
        pygame.draw.line(
            surface,
            track_color,
            (self.rect.left, self.rect.centery),
            (self.rect.right, self.rect.centery),
            4,
        )

        # Calcular posición del handle
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.left + int(ratio * self.rect.width)

        # Línea de progreso
        if ratio > 0:
            pygame.draw.line(
                surface,
                self.config.slider_progress_color,
                (self.rect.left, self.rect.centery),
                (handle_x, self.rect.centery),
                4,
            )

        # Handle del slider
        handle_color = (
            self.config.slider_handle_active
            if self.dragging
            else self.config.slider_handle_color
        )
        pygame.draw.circle(
            surface, handle_color, (handle_x, self.rect.centery), self.handle_radius
        )

        # Borde del handle
        pygame.draw.circle(
            surface,
            self.config.slider_handle_border,
            (handle_x, self.rect.centery),
            self.handle_radius,
            2,
        )

        # Label modernizado
        label = self.font.render(
            f"{self.label}: {int(self.value)}", True, self.config.text_color
        )
        surface.blit(label, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        """Maneja eventos del slider"""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = max(0, min(event.pos[0] - self.rect.left, self.rect.width))
            ratio = rel_x / self.rect.width
            self.value = self.min_val + ratio * (self.max_val - self.min_val)
