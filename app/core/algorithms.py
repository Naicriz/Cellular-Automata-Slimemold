"""Algoritmos optimizados con Numba JIT para el autómata celular"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def activation_vectorized(arr):
    """
    Función de activación optimizada que se paraleliza con cache activado.

    Args:
        arr: Array 2D de valores flotantes

    Returns:
        Array con la función de activación aplicada
    """
    result = np.empty_like(arr)
    for i in prange(arr.shape[0]):  # pylint: disable=not-an-iterable
        for j in prange(arr.shape[1]):  # pylint: disable=not-an-iterable
            x = arr[i, j]
            result[i, j] = -1.0 / (0.89 * x * x + 1.0) + 1.0
    return result


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def enhanced_color_mapping_with_smooth_transitions(normalized):
    """
    Mapeo de colores mejorado con transiciones suaves

    Args:
        normalized: Array 2D normalizado entre 0 y 1

    Returns:
        Array 3D con colores RGB suavizados
    """
    height, width = normalized.shape
    color_array = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            val = normalized[y, x]

            # Aplicar smooth_step para transiciones de color más suaves
            t = max(0.0, min(1.0, val))
            smooth_val = t * t * (3.0 - 2.0 * t)

            # Paleta cálida: negro → rojo intenso → naranja fuerte → dorado → blanco
            if smooth_val <= 0.2:
                # Negro a rojo intenso
                factor = smooth_val / 0.2
                smooth_factor = factor * factor * (3.0 - 2.0 * factor)
                color_array[y, x, 0] = np.uint8(smooth_factor * 180)  # R: 0→180
                color_array[y, x, 1] = np.uint8(smooth_factor * 20)  # G: 0→20
                color_array[y, x, 2] = np.uint8(smooth_factor * 10)  # B: 0→10
            elif smooth_val <= 0.5:
                # Rojo intenso a naranja fuerte
                factor = (smooth_val - 0.2) / 0.3
                smooth_factor = factor * factor * (3.0 - 2.0 * factor)
                color_array[y, x, 0] = np.uint8(180 + smooth_factor * 60)  # R: 180→240
                color_array[y, x, 1] = np.uint8(20 + smooth_factor * 120)  # G: 20→140
                color_array[y, x, 2] = np.uint8(10 + smooth_factor * 30)  # B: 10→40
            else:
                # Naranja fuerte a dorado/blanco
                factor = (smooth_val - 0.5) / 0.5
                smooth_factor = factor * factor * (3.0 - 2.0 * factor)
                color_array[y, x, 0] = np.uint8(240 + smooth_factor * 15)  # R: 240→255
                color_array[y, x, 1] = np.uint8(140 + smooth_factor * 115)  # G: 140→255
                color_array[y, x, 2] = np.uint8(40 + smooth_factor * 215)  # B: 40→255

    return color_array


@jit(nopython=True, fastmath=True, cache=True)
def fast_distance_computation(grid):
    """
    Cálculo optimizado de distancias usando fast_inverse_sqrt aproximado

    Args:
        grid: Grid 2D del autómata

    Returns:
        Grid con valores de distancia computados rápidamente
    """
    height, width = grid.shape
    result = np.zeros_like(grid)
    center_x, center_y = width // 2, height // 2

    for y in range(height):
        for x in range(width):
            # Calcular distancia squared
            dx = x - center_x
            dy = y - center_y
            dist_squared = dx * dx + dy * dy

            if dist_squared > 0:
                # Usar aproximación rápida para 1/sqrt(x)
                # En Numba, usamos la función estándar que está optimizada
                inv_dist = 1.0 / np.sqrt(dist_squared)
                result[y, x] = grid[y, x] * inv_dist * 0.1
            else:
                result[y, x] = grid[y, x]

    return result


@jit(nopython=True, fastmath=True, cache=True)
def optimized_neighbor_influence(grid, influence_radius=3):
    """
    Cálculo optimizado de influencia de vecinos usando fast computation

    Args:
        grid: Grid 2D del autómata
        influence_radius: Radio de influencia

    Returns:
        Grid con influencia de vecinos aplicada
    """
    height, width = grid.shape
    result = np.copy(grid)

    for y in range(height):
        for x in range(width):
            total_influence = 0.0
            neighbor_count = 0

            # Examinar vecinos en el radio especificado
            for dy in range(-influence_radius, influence_radius + 1):
                for dx in range(-influence_radius, influence_radius + 1):
                    if dx == 0 and dy == 0:
                        continue

                    ny = (y + dy) % height  # Wrap around
                    nx = (x + dx) % width

                    # Calcular distancia squared
                    dist_squared = dx * dx + dy * dy

                    if dist_squared <= influence_radius * influence_radius:
                        # Usar fast inverse sqrt approximation
                        if dist_squared > 0:
                            inv_dist = 1.0 / np.sqrt(dist_squared)
                            influence = grid[ny, nx] * inv_dist
                            total_influence += influence
                            neighbor_count += 1

            if neighbor_count > 0:
                avg_influence = total_influence / neighbor_count
                result[y, x] = (grid[y, x] + avg_influence * 0.1) * 0.5

    return result


def precompile_jit_functions():
    """Pre-compila las funciones JIT para optimizar el primer uso"""
    print("⚡ Pre-compilando funciones JIT...")
    dummy = np.random.rand(10, 10).astype(np.float32)
    activation_vectorized(dummy)
    enhanced_color_mapping_with_smooth_transitions(dummy)
    print("✅ JIT compilado y optimizado")
