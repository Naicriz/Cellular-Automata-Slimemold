"""Utilidades matemáticas para el proyecto
aún no se han implementado todas las funciones
"""

import math
from typing import Tuple

import numpy as np


def normalize_array(
    arr: np.ndarray, min_val: float = 0.0, max_val: float = 1.0
) -> np.ndarray:
    """
    Normaliza un array numpy entre min_val y max_val.

    Args:
        arr: Array numpy a normalizar
        min_val: Valor mínimo de salida
        max_val: Valor máximo de salida

    Returns:
        Array normalizado
    """
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max == arr_min:
        return np.full_like(arr, min_val)

    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (max_val - min_val) + min_val


def smooth_step(edge0: float, edge1: float, x: float) -> float:
    """
    Función smoothstep para interpolación suave.

    Args:
        edge0: Borde inferior
        edge1: Borde superior
        x: Valor a interpolar

    Returns:
        Valor interpolado suavemente
    """
    # Clamp x al rango [0, 1]
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    # Usar función smoothstep cúbica
    return t * t * (3.0 - 2.0 * t)


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula la distancia euclidiana entre dos puntos 2D"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def angle_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula el ángulo en radianes entre dos puntos.

    Args:
        p1: Punto inicial (x, y)
        p2: Punto final (x, y)

    Returns:
        Ángulo en radianes
    """
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def gaussian_2d(x: float, y: float, sigma: float = 1.0) -> float:
    """
    Función gaussiana 2D centrada en (0, 0).

    Args:
        x: Coordenada X
        y: Coordenada Y
        sigma: Desviación estándar

    Returns:
        Valor gaussiano
    """
    return math.exp(-(x**2 + y**2) / (2 * sigma**2))


def create_circular_kernel(radius: int, normalize: bool = True) -> np.ndarray:
    """
    Crea un kernel circular para operaciones de convolución.

    Args:
        radius: Radio del kernel
        normalize: Si normalizar el kernel

    Returns:
        Kernel numpy array
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    center = radius

    for i in range(size):
        for j in range(size):
            distance = math.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance <= radius:
                kernel[i, j] = 1.0

    if normalize and kernel.sum() > 0:
        kernel = kernel / kernel.sum()

    return kernel


def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Crea un kernel gaussiano para desenfoque.

    Args:
        size: Tamaño del kernel (debe ser impar)
        sigma: Desviación estándar

    Returns:
        Kernel gaussiano normalizado
    """
    if size % 2 == 0:
        size += 1  # Asegurar que sea impar

    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = gaussian_2d(x, y, sigma)

    # Normalizar
    return kernel / kernel.sum()


def map_range(
    value: float, in_min: float, in_max: float, out_min: float, out_max: float
) -> float:
    """
    Mapea un valor de un rango a otro.

    Args:
        value: Valor a mapear
        in_min: Mínimo del rango de entrada
        in_max: Máximo del rango de entrada
        out_min: Mínimo del rango de salida
        out_max: Máximo del rango de salida

    Returns:
        Valor mapeado
    """
    # Evitar división por cero
    if in_max == in_min:
        return out_min

    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def sigmoid(x: float, steepness: float = 1.0) -> float:
    """
    Función sigmoide para activación suave.

    Args:
        x: Valor de entrada
        steepness: Inclinación de la curva

    Returns:
        Valor entre 0 y 1
    """
    try:
        return 1.0 / (1.0 + math.exp(-steepness * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def periodic_function(
    x: float, period: float = 1.0, amplitude: float = 1.0, phase: float = 0.0
) -> float:
    """
    Función periódica (seno) con parámetros ajustables.

    Args:
        x: Valor de entrada
        period: Período de la función
        amplitude: Amplitud de la función
        phase: Desfase de la función

    Returns:
        Valor de la función periódica
    """
    return amplitude * math.sin(2 * math.pi * x / period + phase)


def fast_inverse_sqrt(x: float) -> float:
    """
    Aproximación rápida de 1/sqrt(x) usando el método de Quake III.

    Args:
        x: Valor de entrada (debe ser positivo)

    Returns:
        Aproximación de 1/sqrt(x)
    """
    if x <= 0:
        return 0.0

    # Para Python, usamos el método estándar que es suficientemente rápido
    return 1.0 / math.sqrt(x)
