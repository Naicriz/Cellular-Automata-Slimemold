"""Configuraciones de la aplicación"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from app.config.constants import DEFAULT_ERASER_SIZE


@dataclass
class SimulationSettings:
    """Configuraciones de la simulación"""

    # Configuración de filtros
    filters: dict[str, np.ndarray] | None = None

    # Configuración de renderizado
    vsync_enabled: bool = True
    show_info: bool = True

    # Configuración de interacción
    initial_eraser_size: int = DEFAULT_ERASER_SIZE
    initial_zoom: float = 1.0
    initial_skip_frames: int = 1

    def __post_init__(self):
        """Inicializa los filtros por defecto si no se proporcionan"""
        if self.filters is None:
            self.filters = {
                "default": np.array(
                    [
                        [0.8, -0.85, 0.8],
                        [-0.85, -0.2, -0.85],
                        [0.8, -0.85, 0.8],
                    ],
                    dtype=np.float32,
                ),
            }


@dataclass
class CacheSettings:
    """Configuraciones del sistema de cache"""

    enable_ui_cache: bool = True
    enable_render_cache: bool = True
    cache_tolerance_zoom: float = 0.5


@dataclass
class PerformanceSettings:
    """Configuraciones de rendimiento"""

    # FPS y rendimiento
    target_fps: int = 60
    adaptive_quality: bool = True
    quality_factor: float = 1.0

    def map_quality_to_render_scale(self) -> int:
        """
        Mapea la calidad a escala de renderizado usando map_range

        Returns:
            Escala de renderizado (1-4)
        """
        from app.utils.math_utils import map_range

        # Mapear calidad (0.0-1.0) a escala de render (1-4)
        # Menor calidad = mayor escala (menos píxeles)
        render_scale = map_range(self.quality_factor, 0.0, 1.0, 4.0, 1.0)
        return max(1, int(render_scale))

    def map_fps_to_skip_frames(self, current_fps: float) -> int:
        """
        Mapea FPS actual a frames a saltar para mantener rendimiento

        Args:
            current_fps: FPS actual medido

        Returns:
            Número de frames a saltar
        """
        from app.utils.math_utils import map_range

        # Si FPS es muy bajo, saltar más frames
        if current_fps < 30:
            return int(map_range(current_fps, 10, 30, 5, 2))
        elif current_fps < 45:
            return int(map_range(current_fps, 30, 45, 2, 1))
        else:
            return 1  # No saltar frames si FPS es bueno


@dataclass
class VisualizationSettings:
    """Configuraciones de visualización"""

    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0

    def map_brightness_to_color_multiplier(self) -> float:
        """
        Mapea brillo del usuario a multiplicador de color interno

        Returns:
            Multiplicador de color (0.1-3.0)
        """
        from app.utils.math_utils import map_range

        # Mapear brillo de usuario (0.0-2.0) a multiplicador interno
        return map_range(self.brightness, 0.0, 2.0, 0.1, 3.0)

    def map_contrast_to_curve_factor(self) -> float:
        """
        Mapea contraste a factor de curva

        Returns:
            Factor de curva de contraste (0.5-2.0)
        """
        from app.utils.math_utils import map_range

        return map_range(self.contrast, 0.0, 2.0, 0.5, 2.0)
