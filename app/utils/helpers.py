"""Funciones auxiliares para el proyecto"""

import time
from typing import Any, Tuple

import pygame


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Limita un valor entre min y max"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Interpolación lineal entre a y b, sirve para suavizar transiciones.
    Args:
        a: Valor inicial
        b: Valor final
        t: Proporción (0.0 a 1.0)
    Returns:
        Valor interpolado
    """
    return a + (b - a) * clamp(t, 0.0, 1.0)


def format_number(num: float, decimals: int = 1) -> str:
    """Formatea un número con decimales específicos
    Args:
        num: Número a formatear
        decimals: Cantidad de decimales a mostrar
    Returns:
        Número formateado como cadena
    """
    if isinstance(num, int):
        return str(num)
    return f"{num:.{decimals}f}"


def get_fps_color(fps: int, target_fps: int = 60) -> Tuple[int, int, int]:
    """
    Retorna un color basado en el FPS actual.

    Args:
        fps: FPS actual
        target_fps: FPS objetivo

    Returns:
        Color RGB como tupla
    """
    ratio = fps / target_fps

    if ratio >= 0.9:  # 90%+ del objetivo
        return (100, 255, 100)  # Verde
    elif ratio >= 0.7:  # 70%+ del objetivo
        return (255, 255, 100)  # Amarillo
    else:  # Menos del 70%
        return (255, 100, 100)  # Rojo


def create_gradient_surface(
    width: int,
    height: int,
    start_color: Tuple[int, int, int],
    end_color: Tuple[int, int, int],
    vertical: bool = True,
) -> pygame.Surface:
    """
    Crea una superficie con gradiente.

    Args:
        width: Ancho de la superficie
        height: Alto de la superficie
        start_color: Color inicial RGB
        end_color: Color final RGB
        vertical: Si True, gradiente vertical; si False, horizontal

    Returns:
        Superficie pygame con gradiente
    """
    surface = pygame.Surface((width, height))

    if vertical:
        for y in range(height):
            ratio = y / height
            color = (
                int(lerp(start_color[0], end_color[0], ratio)),
                int(lerp(start_color[1], end_color[1], ratio)),
                int(lerp(start_color[2], end_color[2], ratio)),
            )
            pygame.draw.line(surface, color, (0, y), (width, y))
    else:
        for x in range(width):
            ratio = x / width
            color = (
                int(lerp(start_color[0], end_color[0], ratio)),
                int(lerp(start_color[1], end_color[1], ratio)),
                int(lerp(start_color[2], end_color[2], ratio)),
            )
            pygame.draw.line(surface, color, (x, 0), (x, height))

    return surface


def safe_font_load(font_name: str, size: int, bold: bool = False) -> pygame.font.Font:
    """
    Carga una fuente de forma segura con fallback.

    Args:
        font_name: Nombre de la fuente
        size: Tamaño de la fuente
        bold: Si la fuente debe ser negrita

    Returns:
        Objeto de fuente pygame
    """
    try:
        return pygame.font.SysFont(font_name, size, bold=bold)
    except (pygame.error, OSError):
        return pygame.font.Font(None, size)


def measure_time(func):
    """Decorador para medir tiempo de ejecución de funciones
    Args:
        func: Función a medir
    Returns:
        Función envuelta que imprime el tiempo de ejecución
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__}: {(end_time - start_time) * 1000:.2f}ms")
        return result

    return wrapper


def create_text_surface(
    text: str,
    font: pygame.font.Font,
    color: Tuple[int, int, int],
    background: Tuple[int, int, int] | None = None,
) -> pygame.Surface:
    """
    Crea una superficie de texto con fondo opcional.

    Args:
        text: Texto a renderizar
        font: Fuente pygame
        color: Color del texto RGB
        background: Color de fondo RGB (opcional)

    Returns:
        Superficie pygame con el texto
    """
    if background:
        return font.render(text, True, color, background)
    else:
        return font.render(text, True, color)


def calculate_eraser_area(
    center_x: int, center_y: int, radius: int, grid_size: int
) -> list:
    """Calcula las coordenadas afectadas por el borrador circular"""
    from app.utils.math_utils import distance_2d

    affected_cells = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            distance = distance_2d((center_x + dx, center_y + dy), (center_x, center_y))

            if distance <= radius:
                new_x = center_x + dx
                new_y = center_y + dy

                if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                    fade_factor = 1.0 - (distance / radius) if radius > 0 else 1.0
                    affected_cells.append((new_x, new_y, fade_factor))

    return affected_cells
