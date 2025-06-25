"""Nuevo main.py reestructurado - Punto de entrada principal

Automatización de Slime Molds con arquitectura modular:
- Separación clara de responsabilidades
- Componentes reutilizables
- Fácil mantenimiento y extensión
"""

import sys
from pathlib import Path

import pygame

# Agregar la ruta del proyecto al PYTHONPATH para importaciones relativas
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Intentar importación desde raíz del proyecto
    from app.config.constants import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH
    from app.core.algorithms import precompile_jit_functions
    from app.core.automata import UltraSlimeMold
except ImportError:
    # Fallback para ejecución desde dentro de app/
    from config.constants import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH
    from core.algorithms import precompile_jit_functions
    from core.automata import UltraSlimeMold


def initialize_pygame():
    """Inicializa Pygame con configuraciones básicas"""
    pygame.init()
    pygame.display.set_caption("Ultra Slime Mold Automata - Modular")

    # Configurar pantalla
    try:
        screen = pygame.display.set_mode(
            (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT),
            pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE,
        )
        return screen
    except pygame.error as e:
        print(f"Error al inicializar Pygame: {e}")
        return None


def main():
    """Función principal del programa"""
    print("🚀 Iniciando Ultra Slime Mold Automata - Versión Modular")

    # Inicializar Pygame
    screen = initialize_pygame()
    if not screen:
        print("❌ Error al inicializar la ventana")
        sys.exit(1)

    # Pre-compilar funciones JIT
    precompile_jit_functions()

    # Crear y ejecutar el autómata
    try:
        automata = UltraSlimeMold(screen)
        automata.run()
    except KeyboardInterrupt:
        print("\n⏹️ Simulación interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        raise
    finally:
        pygame.quit()
        print("👋 ¡Hasta luego!")


if __name__ == "__main__":
    main()
