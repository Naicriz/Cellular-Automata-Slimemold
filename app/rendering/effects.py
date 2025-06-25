"""Efectos visuales y filtros para el renderizado"""

import numpy as np
from scipy import ndimage


class EffectsProcessor:
    """Procesador de efectos visuales para el autómata"""

    def __init__(self):
        # Filtros predefinidos para la convolución
        self.filters = {
            "default": np.array(
                [[0.8, -0.85, 0.8], [-0.85, -0.2, -0.85], [0.8, -0.85, 0.8]],
                dtype=np.float32,
            ),
            "smooth": np.array(
                [[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32
            ),
            "edge": np.array(
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
            ),
        }

    def get_filter(self, filter_name):
        """Obtiene un filtro por nombre"""
        return self.filters.get(filter_name, self.filters["default"])

    def add_filter(self, name, filter_matrix):
        """Agrega un nuevo filtro personalizado"""
        self.filters[name] = np.array(filter_matrix, dtype=np.float32)

    def list_filters(self):
        """Lista todos los filtros disponibles"""
        return list(self.filters.keys())

    def apply_glow_effect(self, grid, intensity=0.1):
        """Aplica un efecto de brillo al grid"""
        # Crear efecto de glow expandiendo valores altos
        glowed = grid + (grid**2) * intensity
        return np.clip(glowed, 0.0, 1.0)

    def apply_contrast(self, grid, factor=1.2):
        """Aplica contraste al grid"""
        # Aumentar contraste
        contrasted = (grid - 0.5) * factor + 0.5
        return np.clip(contrasted, 0.0, 1.0)

    def apply_blur(self, grid, radius=1):
        """Aplica desenfoque gaussiano simple"""
        # Usar filtro gaussiano básico para desenfoque
        return ndimage.gaussian_filter(grid, sigma=radius)
