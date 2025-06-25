"""Sistema de renderizado principal"""

import numpy as np
import pygame

from ..config.constants import GRID_DISPLAY_HEIGHT, GRID_DISPLAY_WIDTH
from .colors import ColorMapper
from .effects import EffectsProcessor


class Renderer:
    """Sistema de renderizado optimizado para el autómata"""

    def __init__(self):
        self.color_mapper = ColorMapper()
        self.effects_processor = EffectsProcessor()

        # Cache de superficies
        self.surface_cache = {}
        self.last_grid_hash = None
        self.cached_surface = None

    def render_grid_to_surface(
        self, grid, render_scale=1, zoom_factor=1.0, zoom_offset_x=0, zoom_offset_y=0
    ):
        """
        Renderiza el grid a una superficie pygame.

        Args:
            grid: Grid 2D del autómata
            render_scale: Escala de renderizado (>1 para downsampling)
            zoom_factor: Factor de zoom
            zoom_offset_x: Offset horizontal del zoom
            zoom_offset_y: Offset vertical del zoom

        Returns:
            Superficie pygame renderizada
        """
        # Crear superficie coloreada
        color_array = self.color_mapper.create_surface_from_grid(grid, render_scale)

        if color_array.shape[0] == 0 or color_array.shape[1] == 0:
            return pygame.Surface((GRID_DISPLAY_WIDTH, GRID_DISPLAY_HEIGHT))

        # Crear superficie pygame desde el array
        color_surface = pygame.surfarray.make_surface(color_array.swapaxes(0, 1))

        # Escalar a tamaño objetivo
        target_width = GRID_DISPLAY_WIDTH
        target_height = GRID_DISPLAY_HEIGHT

        if color_surface.get_size() != (target_width, target_height):
            color_surface = pygame.transform.scale(
                color_surface, (target_width, target_height)
            )

        # Aplicar zoom si es necesario
        if zoom_factor > 1.0:
            zoomed_width = int(target_width * zoom_factor)
            zoomed_height = int(target_height * zoom_factor)
            color_surface = pygame.transform.scale(
                color_surface, (zoomed_width, zoomed_height)
            )

            # Crear superficie recortada según el offset
            visible_rect = pygame.Rect(
                zoom_offset_x, zoom_offset_y, target_width, target_height
            )
            color_surface = color_surface.subsurface(visible_rect)

        return color_surface

    def create_optimized_surface(self, grid, **kwargs):
        """Crea superficie con optimizaciones de cache"""
        # Calcular hash simple del grid para cache
        grid_hash = hash(grid.data.tobytes()[::1000])

        # Verificar cache
        cache_key = (
            grid_hash,
            kwargs.get("render_scale", 1),
            kwargs.get("zoom_factor", 1.0),
        )

        if cache_key in self.surface_cache:
            return self.surface_cache[cache_key]

        # Crear nueva superficie
        surface = self.render_grid_to_surface(grid, **kwargs)

        # Guardar en cache (limitado a 10 entradas)
        if len(self.surface_cache) > 15:
            # Limpiar cache más antiguo
            oldest_key = next(iter(self.surface_cache))
            del self.surface_cache[oldest_key]

        self.surface_cache[cache_key] = surface
        return surface

    def set_color_palette(self, palette_name):
        """Cambia la paleta de colores"""
        return self.color_mapper.set_palette(palette_name)

    def get_available_palettes(self):
        """Obtiene las paletas disponibles"""
        return self.color_mapper.get_available_palettes()

    def clear_cache(self):
        """Limpia el cache de superficies"""
        self.surface_cache.clear()
        self.last_grid_hash = None
        self.cached_surface = None
