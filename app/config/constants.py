"""Constantes del proyecto Slime Mold Automata"""

# Dimensiones de ventana
DEFAULT_WINDOW_WIDTH = 1700
DEFAULT_WINDOW_HEIGHT = 1000

# Configuración del grid
GRID_DISPLAY_WIDTH = 1000
GRID_DISPLAY_HEIGHT = 1000
GRID_SIZE = 1000
CELL_SIZE = GRID_DISPLAY_WIDTH // GRID_SIZE

# Configuración de rendimiento
FPS = 60
RENDER_SCALE = 1


# Colores de la interfaz - Paleta modernizada
class UIColors:
    # Colores base
    BACKGROUND = (0, 0, 0)

    # Botones
    BUTTON_NORMAL = (32, 34, 37)  # Dark slate
    BUTTON_HOVER = (44, 47, 51)  # Slightly lighter
    BUTTON_ACTIVE = (88, 101, 242)  # Blurple (Discord-style)

    # Switches
    SWITCH_ON = (0, 200, 140)  # Neo-mint green
    SWITCH_OFF = (78, 80, 85)  # Dark gray

    # Texto
    TEXT_PRIMARY = (240, 240, 255)  # Off-white
    TEXT_SECONDARY = (200, 200, 200)
    TEXT_SUCCESS = (220, 255, 200)

    # Sliders
    SLIDER_TRACK = (45, 48, 55)
    SLIDER_TRACK_HOVER = (60, 65, 75)
    SLIDER_HANDLE = (0, 200, 140)  # Neo-mint
    SLIDER_HANDLE_ACTIVE = (0, 230, 160)
    SLIDER_HANDLE_BORDER = (85, 90, 95)
    SLIDER_PROGRESS = (88, 101, 242)  # Blurple
    SLIDER_LABEL = (230, 240, 255)  # Casi blanco


# Configuración de zoom
MIN_ZOOM = 1.0
MAX_ZOOM = 6.0

# Tamaños de menú
MENU_WIDTH_RATIO = 0.20  # 20% del ancho de ventana
HEADER_HEIGHT = 40
CONTROL_HEIGHT = 30
CONTROL_SPACING = 35
SECTION_SPACING = 15

# Configuración del borrador
DEFAULT_ERASER_SIZE = 10
MIN_ERASER_SIZE = 5
MAX_ERASER_SIZE = 30
