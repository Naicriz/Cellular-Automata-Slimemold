"""Mapeo de colores para la visualización"""

import numpy as np

from app.core.algorithms import enhanced_color_mapping_with_smooth_transitions
from app.utils.math_utils import normalize_array, smooth_step


class ColorMapper:
    """Maneja el mapeo de colores para la visualización del autómata"""

    def __init__(self):
        self.current_palette = "bioluminescent"

    def create_surface_from_grid(self, grid, render_scale=1):
        """
        Crea una superficie coloreada a partir del grid.

        Args:
            grid: Grid 2D del autómata
            render_scale: Escala de renderizado (>1 para downsampling)

        Returns:
            Array de colores RGB
        """
        # Downsample si es necesario para optimizar rendimiento
        if render_scale > 1:
            downsampled = grid[::render_scale, ::render_scale]
        else:
            downsampled = grid

        # Normalizar valores para mejor mapeo de colores con rango personalizable
        normalized = normalize_array(downsampled, 0.0, 1.0)

        # Usar función JIT optimizada para mapeo de colores
        return enhanced_color_mapping_with_smooth_transitions(normalized)

    def get_available_palettes(self):
        """Retorna las paletas disponibles"""
        return ["bioluminescent", "heat", "cool"]

    def set_palette(self, palette_name):
        """Cambia la paleta de colores"""
        if palette_name in self.get_available_palettes():
            self.current_palette = palette_name
            return True
        return False

    def enhance_contrast(self, grid, intensity=1.0):
        """
        Mejora el contraste del grid usando normalización personalizada

        Args:
            grid: Grid 2D del autómata
            intensity: Intensidad del realce (0.0 a 2.0)

        Returns:
            Grid con contraste mejorado
        """

        # Normalizar primero
        normalized = normalize_array(grid, 0.0, 1.0)

        # Aplicar realce de contraste usando smooth_step
        enhanced = np.zeros_like(normalized)
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                val = normalized[i, j]
                enhanced[i, j] = smooth_step(0.0, 1.0, val) * intensity

        return np.clip(enhanced, 0.0, 1.0)

    def create_heat_map(self, grid, temperature_range=(0.0, 1.0)):
        """
        Crea un mapa de calor normalizado

        Args:
            grid: Grid 2D del autómata
            temperature_range: Rango de temperatura (min, max)

        Returns:
            Grid normalizado para visualización térmica
        """
        min_temp, max_temp = temperature_range
        return normalize_array(grid, min_temp, max_temp)
