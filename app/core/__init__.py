"""Módulo core del proyecto"""

from .algorithms import (
    activation_vectorized,
    enhanced_color_mapping_with_smooth_transitions,
    precompile_jit_functions,
)
from .automata import UltraSlimeMold

__all__ = [
    "UltraSlimeMold",
    "activation_vectorized",
    "enhanced_color_mapping_with_smooth_transitions",
    "precompile_jit_functions",
]
