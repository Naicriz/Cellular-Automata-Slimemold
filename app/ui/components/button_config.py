"""Configuración de botones para la interfaz"""

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class ButtonConfig:
    """Configuración de un botón del menú"""

    x: int
    y: int
    width: int
    height: int
    text: str
    action: Callable
    is_switch: bool = False
    switch_state: bool = False

    # Paleta modernizada
    color_normal: Tuple[int, int, int] | None = (32, 34, 37)  # Dark slate
    color_hover: Tuple[int, int, int] | None = (44, 47, 51)  # Slightly lighter
    color_active: Tuple[int, int, int] | None = (88, 101, 242)  # Blurple

    color_switch_on: Tuple[int, int, int] | None = (0, 200, 140)  # Neo-mint green
    color_switch_off: Tuple[int, int, int] | None = (78, 80, 85)  # Dark gray

    text_color: Tuple[int, int, int] | None = (240, 240, 255)  # Off-white
    font_size: int = 20

    slider_track_color: Tuple[int, int, int] | None = (45, 48, 55)
    slider_track_hover: Tuple[int, int, int] | None = (60, 65, 75)
    slider_handle_color: Tuple[int, int, int] | None = (0, 200, 140)  # Neo-mint
    slider_handle_active: Tuple[int, int, int] | None = (0, 230, 160)
    slider_handle_border: Tuple[int, int, int] | None = (85, 90, 95)
    slider_progress_color: Tuple[int, int, int] | None = (88, 101, 242)  # Blurple
    slider_label_color: Tuple[int, int, int] | None = (230, 240, 255)  # Casi blanco
