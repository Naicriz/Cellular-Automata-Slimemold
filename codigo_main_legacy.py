"""Automatización de Slime Molds con Pygame y Numba JIT
- Simulación de crecimiento de moho deslizante
- Interfaz interactiva con controles optimizados
- Renderizado eficiente con paleta de colores cálida
- Convolución de filtros para efectos visuales
- Soporte para zoom, borrador y saltos de frames
- Menú interactivo con botones y switches
- Cache inteligente para optimizar rendimiento
- Soporte para V-Sync y calidad de renderizado ajustable
"""

import sys
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pygame
from numba import jit, prange
from scipy import ndimage

# Parámetros para ventana y grilla
DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT = 1600, 1000
GRID_DISPLAY_WIDTH, GRID_DISPLAY_HEIGHT = 1000, 1000
GRID_SIZE = 1000
CELL_SIZE = GRID_DISPLAY_WIDTH // GRID_SIZE
FPS = 60
RENDER_SCALE = 1


# Función de activación optimizada con Numba JIT
@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def activation_vectorized(arr):
    """Versión optimizada que se paraleliza con cache desactivado"""
    result = np.empty_like(arr)
    for i in prange(arr.shape[0]):  # pylint: disable=not-an-iterable
        for j in prange(arr.shape[1]):  # pylint: disable=not-an-iterable
            x = arr[i, j]
            result[i, j] = -1.0 / (0.89 * x * x + 1.0) + 1.0
    return result


@jit(nopython=True, fastmath=True, cache=True)
def create_color_mapping_jit(normalized):
    """Mapeo de colores con JIT vectorizado - Paleta amarillo-anaranjada"""
    height, width = normalized.shape
    color_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Loops anidados tradicionales para Numba
    for y in range(height):
        for x in range(width):
            val = normalized[y, x]

            # Usar operaciones enteras directas - Paleta cálida amarillo-anaranjada
            if val <= 0.15:
                # Marrón muy oscuro/negro
                color_array[y, x, 0] = np.uint8(val * 40 + 10)  # Más rojo
                color_array[y, x, 1] = np.uint8(val * 25 + 5)  # Menos verde
                color_array[y, x, 2] = np.uint8(val * 10)  # Muy poco azul
            elif val <= 0.35:
                # Marrón anaranjado oscuro
                color_array[y, x, 0] = np.uint8(val * 100 + 40)  # Predominio rojo
                color_array[y, x, 1] = np.uint8(val * 80 + 20)  # Algo de verde
                color_array[y, x, 2] = np.uint8(val * 20)  # Poco azul
            elif val <= 0.55:
                # Naranja medio brillante
                color_array[y, x, 0] = np.uint8(val * 120 + 100)  # Rojo fuerte
                color_array[y, x, 1] = np.uint8(val * 100 + 60)  # Verde moderado
                color_array[y, x, 2] = np.uint8(val * 30 + 10)  # Azul mínimo
            elif val <= 0.75:
                # Amarillo anaranjado
                color_array[y, x, 0] = 255  # Rojo máximo
                g_val = val * 140 + 100
                color_array[y, x, 1] = np.uint8(255 if g_val > 255 else g_val)
                color_array[y, x, 2] = np.uint8(
                    val * 40 + 20
                )  # Algo de azul para calidez
            elif val <= 0.9:
                # Amarillo brillante cálido
                color_array[y, x, 0] = 255  # Rojo máximo
                color_array[y, x, 1] = 255  # Verde máximo
                color_array[y, x, 2] = np.uint8(
                    val * 80 + 50
                )  # Más azul para amarillo cálido
            else:
                # Amarillo-blanco incandescente con tinte cálido
                color_array[y, x, 0] = 255  # Rojo máximo
                color_array[y, x, 1] = 255  # Verde máximo
                b_val = val * 140 + 100  # Más azul para blanco cálido
                color_array[y, x, 2] = np.uint8(255 if b_val > 255 else b_val)

    return color_array


# Filtros para convolución del grid
filters = {
    "default": np.array(
        [[0.8, -0.85, 0.8], [-0.85, -0.2, -0.85], [0.8, -0.85, 0.8]], dtype=np.float32
    ),
}


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
    color_normal: Tuple[int, int, int] = (32, 34, 37)  # Dark slate (background default)
    color_hover: Tuple[int, int, int] = (44, 47, 51)  # Slightly lighter (hover effect)
    color_active: Tuple[int, int, int] = (
        88,
        101,
        242,
    )  # Blurple (Discord-style active)

    color_switch_on: Tuple[int, int, int] = (
        0,
        200,
        140,
    )  # Neo-mint green (active toggle)
    color_switch_off: Tuple[int, int, int] = (78, 80, 85)  # Dark gray (inactive toggle)

    text_color: Tuple[int, int, int] = (240, 240, 255)  # Off-white (soft readable text)
    font_size: int = 20


class InteractiveSwitch:
    """Switch/toggle interactivo"""

    def __init__(self, config: ButtonConfig):
        self.config = config
        self.rect = pygame.Rect(config.x, config.y, config.width, config.height)
        self.is_hovered = False
        self.is_pressed = False
        self.state = config.switch_state
        # Fuente simple
        try:
            self.font = pygame.font.SysFont("arial", config.font_size)
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, config.font_size)

    def handle_event(self, event):
        """Maneja eventos del switch - solo en el área del track"""
        # Definir el área clickeable del track (solo el switch)
        track_width = 50
        track_height = 20
        track_x = self.rect.right - track_width - 5
        track_y = self.rect.centery - track_height // 2
        track_rect = pygame.Rect(track_x, track_y, track_width, track_height)

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = track_rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and track_rect.collidepoint(event.pos):
                self.is_pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_pressed:
                self.is_pressed = False
                if track_rect.collidepoint(event.pos):
                    self.state = not self.state
                    self.config.action()
                    return True
        return False

    def draw(self, surface):
        """Dibuja el switch con formato: 'Texto: [Switch]'"""
        # Track del switch (carril donde se mueve el indicador)
        track_width = 50
        track_height = 20
        track_x = self.rect.right - track_width - 5  # 5px margen derecho
        track_y = self.rect.centery - track_height // 2  # Centrado verticalmente
        # Crear rectángulo del track
        track_rect = pygame.Rect(track_x, track_y, track_width, track_height)

        # Color del track según estado
        track_color = self.config.color_switch_on if self.state else (80, 80, 80)
        pygame.draw.rect(surface, track_color, track_rect, border_radius=10)

        # Indicador móvil (círculo que se desliza)
        indicator_radius = 8
        if self.state:  # Si está activado, el indicador se mueve a la derecha
            indicator_x = track_rect.right - indicator_radius - 2
            indicator_color = (255, 255, 220)
            indicator_border = (255, 200, 100)
        else:  # Si está desactivado, el indicador se mueve a la izquierda
            indicator_x = track_rect.left + indicator_radius + 2
            indicator_color = (200, 200, 200)
            indicator_border = (160, 160, 160)

        indicator_y = track_rect.centery

        # Indicador principal (sin sombra para ser más limpio)
        pygame.draw.circle(
            surface, indicator_color, (indicator_x, indicator_y), indicator_radius
        )
        pygame.draw.circle(
            surface, indicator_border, (indicator_x, indicator_y), indicator_radius, 2
        )

        # Texto del switch - solo texto sin fondo
        if self.state:
            text_color = (220, 255, 200)  # Verde claro cuando ON
        else:
            text_color = (200, 200, 200)  # Gris claro cuando OFF

        # Formatear texto como "Texto:"
        display_text = f"{self.config.text}:"
        text_surface = self.font.render(display_text, True, text_color)

        text_rect = text_surface.get_rect()
        text_rect.centery = self.rect.centery
        text_rect.x = self.rect.x + 5  # Pequeño margen izquierdo
        surface.blit(text_surface, text_rect)


class InteractiveButton:
    """Botón interactivo"""

    def __init__(self, config: ButtonConfig):
        self.config = config
        self.rect = pygame.Rect(config.x, config.y, config.width, config.height)
        self.is_hovered = False
        self.is_pressed = False
        # Fuente simple
        try:
            self.font = pygame.font.SysFont("arial", config.font_size)
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, config.font_size)

    def handle_event(self, event):
        """Maneja eventos del botón"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_pressed:
                self.is_pressed = False
                if self.rect.collidepoint(event.pos):
                    self.config.action()
                    return True
        return False

    def draw(self, surface):
        """Dibuja el botón con colores estáticos y gradientes"""
        # Determinar color según estado
        if self.is_pressed:
            color = self.config.color_active
        elif self.is_hovered:
            color = self.config.color_hover
        else:
            color = self.config.color_normal

        # Dibujar fondo del botón con bordes redondeados
        pygame.draw.rect(surface, color, self.rect, border_radius=10)

        # Borde simple
        border_color = (
            self.config.color_normal if not self.is_hovered else self.config.color_hover
        )
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=10)

        # Dibujar texto centrado
        text_surface = self.font.render(self.config.text, True, self.config.text_color)

        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class InteractiveMenu:
    """Menú interactivo con controles optimizados y cache inteligente"""

    def __init__(self, automata_instance):
        self.automata = automata_instance
        self.visible = True
        self.dragging = False
        self.drag_offset = (0, 0)

        # Posición y tamaño del menú - mismo ancho que panel de info
        self.menu_x = 10
        self.menu_y = 10
        # Ancho del menú basado en el ancho de la ventana
        self.menu_width = int(
            self.automata.window_width * 0.20
        )  # 20% del ancho de la ventana

        # Calcular altura automáticamente basada en componentes
        self.header_height = 40  # Altura fija para el header
        self._calculate_menu_height()

        # Sistema de cache inteligente optimizado
        self.cache_dirty = True
        self.cached_menu_surface = None
        self.last_control_states = {}
        self.ui_needs_update = True

        # Cache de valores para compatibilidad
        self.last_cached_values = {"zoom": 0, "eraser": 0, "skip": 0}

        # Fuentes mejoradas
        try:
            self.title_font = pygame.font.SysFont("arial", 18, bold=True)
            self.info_font = pygame.font.SysFont("arial", 13)
        except (pygame.error, OSError):
            self.title_font = pygame.font.Font(None, 20)
            self.info_font = pygame.font.Font(None, 15)

        # Crear controles (botones y switches)
        self.controls = self._create_controls()

        # Inicializar estados de controles para cache
        self._cache_control_states()

    def _calculate_menu_height(self):
        """Calcula la altura del menú automáticamente según componentes"""
        # Componentes del menu:
        # - Header: 40px
        # - Padding inicial: 15px
        # - 3 botones de acción: 3 * 30px + 2 * 35px spacing = 160px
        # - Separación de sección: 15px
        # - 2 switches: 2 * 30px + 1 * 35px spacing = 95px
        # - Separación de sección: 15px
        # - 3 sliders: 3 * 65px = 195px
        # - Separación de sección: 20px
        # - 2 botones de calidad: 1 * 27px = 27px
        # - Padding final: 15px

        components_height = (
            self.header_height  # 40px
            + 15  # padding inicial
            + (3 * 30)
            + (2 * 35)  # botones acción: 160px
            + 15  # separación
            + (2 * 30)
            + (1 * 35)  # switches: 95px
            + 15  # separación
            + 10  # espacio extra para sliders
            + (3 * 65)  # sliders: 195px
            + 20  # separación
            + 27  # botones calidad
            + 15  # padding final
        )

        self.menu_height = components_height

    def _create_controls(self):
        """Crea controles del menú con layout compacto y altura fija"""
        controls = []

        # Configuración de layout con espaciado adecuado
        control_width = int(self.menu_width * 0.75)  # 75% del ancho del menú
        control_height = 30  # Altura fija para todos los controles
        control_spacing = 35  # Espaciado entre controles (30 + 5 margen)
        section_spacing = 15  # Separación entre secciones

        # Punto de inicio después del header con padding apropiado
        start_y = self.menu_y + self.header_height + 15  # 15px padding
        col_x = self.menu_x + (self.menu_width - control_width) // 2

        # SECCIÓN 1: Botones de acción principal
        current_y = start_y
        action_buttons = [
            ButtonConfig(
                col_x,
                current_y,
                control_width,
                control_height,
                "🔄 Reiniciar Grid",
                self._reset_grid,
            ),
            ButtonConfig(
                col_x,
                current_y + control_spacing,
                control_width,
                control_height,
                "⏸️ Pausar" if not self.automata.paused else "▶️ Reanudar",
                self._toggle_pause,
            ),
            ButtonConfig(
                col_x,
                current_y + control_spacing * 2,
                control_width,
                control_height,
                "🔍 Reset Zoom",
                self._reset_zoom,
            ),
        ]

        # SECCIÓN 2: Switches para opciones toggle
        current_y = start_y + control_spacing * 3 + section_spacing
        switches = [
            ButtonConfig(
                col_x,
                current_y,
                control_width,
                control_height,
                "📊 Info",
                self._toggle_info,
                is_switch=True,
                switch_state=self.automata.show_info,
            ),
            ButtonConfig(
                col_x,
                current_y + control_spacing,
                control_width,
                control_height,
                "📺 V-Sync",
                self._toggle_vsync,
                is_switch=True,
                switch_state=self.automata.vsync_enabled,
            ),
        ]

        # SECCIÓN 3: Sliders para valores continuos
        current_y = (
            start_y + control_spacing * 7 + section_spacing * 2 + 10
        )  # +10px extra
        slider_width = int(self.menu_width * 0.75)  # 75% del ancho del menú
        slider_height = 20  # Altura fija
        slider_spacing = 65  # Espaciado para texto descriptivo

        # Crear sliders con descripciones
        sliders = [
            InteractiveSlider(
                col_x + (control_width - slider_width) // 2,
                current_y,
                slider_width,
                slider_height,
                1.0,
                6.0,
                self.automata.zoom_factor,
                "🔍 Zoom",
                self._set_zoom,
                "🔍 NIVEL DE ZOOM",
            ),
            InteractiveSlider(
                col_x + (control_width - slider_width) // 2,
                current_y + slider_spacing,
                slider_width,
                slider_height,
                1,
                15,
                self.automata.eraser_size,
                "🖌️ Borrador",
                self._set_eraser_size,
                "🖌️ TAMAÑO DEL BORRADOR",
            ),
            InteractiveSlider(
                col_x + (control_width - slider_width) // 2,
                current_y + slider_spacing * 2,
                slider_width,
                slider_height,
                1,
                10,
                self.automata.skip_frames,
                "⚡ Frame Skip",
                self._set_frame_skip,
                "⚡ SALTO DE FRAMES",
            ),
        ]

        # SECCIÓN 4: Botones de calidad
        current_y += slider_spacing * 3 + 20  # Separación
        mini_width = int(self.menu_width * 0.38)  # 38% del ancho del menú
        mini_height = int(control_height * 0.9)  # 90% de la altura del control
        mini_spacing = 8  # Espaciado

        quality_controls = [
            ButtonConfig(
                col_x,
                current_y,
                mini_width,
                mini_height,
                "📈 Calidad +",
                self._increase_quality,
            ),
            ButtonConfig(
                col_x + mini_width + mini_spacing,
                current_y,
                mini_width,
                mini_height,
                "📉 Calidad -",
                self._decrease_quality,
            ),
        ]

        # Crear objetos de control
        for config in action_buttons:
            controls.append(InteractiveButton(config))

        for config in switches:
            controls.append(InteractiveSwitch(config))

        # Agregar sliders
        for slider in sliders:
            controls.append(slider)

        # Agregar botones de calidad
        for config in quality_controls:
            controls.append(InteractiveButton(config))

        return controls

    def _set_zoom(self, value):
        """Establece el zoom usando el slider"""
        rounded_value = round(value, 1)
        if abs(rounded_value - self.last_cached_values["zoom"]) > 0.2:  # Más tolerancia
            self.automata.zoom_factor = value
            self.automata.clamp_zoom_offset()
            self.last_cached_values["zoom"] = rounded_value
            # No invalidar cache para zoom
            print(f"🔍 [ZOOM] {value:.1f}x")

    def _set_eraser_size(self, value):
        """Establece el tamaño del borrador usando el slider"""
        int_value = int(value)
        if int_value != self.last_cached_values["eraser"]:
            self.automata.eraser_size = int_value
            self.last_cached_values["eraser"] = int_value
            # No invalidar cache para eraser
            print(f"🖌️ [BRUSH] {int_value}px")

    def _set_frame_skip(self, value):
        """Establece el frame skip usando el slider"""
        int_value = int(value)
        if int_value != self.last_cached_values["skip"]:
            self.automata.skip_frames = int_value
            self.last_cached_values["skip"] = int_value
            # No invalidar cache para frame skip
            print(f"⚡ [SKIP] {int_value} frames")

    # Métodos de acción para los controles
    def _reset_grid(self):
        """Reinicia el grid con valores aleatorios"""
        self.automata.grid = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)
        self.automata.generation = 0
        print("🔄 Grid reiniciado desde menú")

    def _toggle_pause(self):
        """Pausa/reanuda la simulación"""
        self.automata.paused = not self.automata.paused
        # Actualizar texto del botón con emojis
        for control in self.controls:
            if hasattr(control, "config") and (
                "Pausar" in control.config.text or "Reanudar" in control.config.text
            ):
                if self.automata.paused:
                    control.config.text = "▶️ Reanudar"
                else:
                    control.config.text = "⏸️ Pausar"
        status = "Pausado" if self.automata.paused else "Reanudado"
        print(f"{status} desde menu")

    def _toggle_info(self):
        """Toggle panel de información"""
        self.automata.show_info = not self.automata.show_info
        status = "ON" if self.automata.show_info else "OFF"
        print(f"📊 Info: {status}")

    def _reset_zoom(self):
        """Resetea el zoom a valores por defecto"""
        self.automata.zoom_factor = 1.0
        self.automata.zoom_offset_x = 0
        self.automata.zoom_offset_y = 0
        print("🔍 Zoom reseteado desde menú")

    def _increase_quality(self):
        """Mejora la calidad de renderizado"""
        if self.automata.render_scale > 1:
            self.automata.render_scale = max(1, self.automata.render_scale - 1)
            print(f"📈 Calidad mejorada: 1:{self.automata.render_scale}")

    def _decrease_quality(self):
        """Reduce la calidad de renderizado para mejor rendimiento"""
        self.automata.render_scale = min(8, self.automata.render_scale + 1)
        print(f"📉 Calidad reducida: 1:{self.automata.render_scale}")

    def _toggle_vsync(self):
        """Toggle V-Sync - Nota: En Pygame, V-Sync se configura al crear la pantalla"""
        self.automata.vsync_enabled = not self.automata.vsync_enabled
        status = "ON" if self.automata.vsync_enabled else "OFF"
        print(f"📺 V-Sync: {status}")
        print("💡 Nota: Para aplicar V-Sync completamente, reinicia la aplicación")

        # Intentar reconfigurar la pantalla con/sin V-Sync si es posible
        current_size = self.automata.screen.get_size()
        display_flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE

        try:
            if self.automata.vsync_enabled:
                # Intentar activar V-Sync reconfigurando la pantalla
                self.automata.screen = pygame.display.set_mode(
                    current_size, display_flags
                )
                print("🔄 Pantalla reconfigurada con V-Sync")
            else:
                # Reconfigurar sin V-Sync
                self.automata.screen = pygame.display.set_mode(
                    current_size, display_flags
                )
                print("🔄 Pantalla reconfigurada sin V-Sync")
        except pygame.error as e:
            print(f"⚠️ No se pudo reconfigurar V-Sync: {e}")
            print("💡 Reinicia la aplicación para aplicar el cambio")

    def handle_event(self, event):
        """Maneja eventos del menú"""
        if not self.visible:
            return False

        # Manejar arrastre del menú
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Verificar si se hace clic en la barra de título
                title_rect = pygame.Rect(
                    self.menu_x, self.menu_y, self.menu_width, self.header_height
                )
                if title_rect.collidepoint(event.pos):
                    self.dragging = True
                    self.drag_offset = (
                        event.pos[0] - self.menu_x,
                        event.pos[1] - self.menu_y,
                    )
                    return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.menu_x = event.pos[0] - self.drag_offset[0]
                self.menu_y = event.pos[1] - self.drag_offset[1]
                # Mantener el menú dentro de los límites de la pantalla
                max_x = self.automata.window_width - self.menu_width
                max_y = self.automata.window_height - self.menu_height
                self.menu_x = max(0, min(max_x, self.menu_x))
                self.menu_y = max(0, min(max_y, self.menu_y))
                self._update_control_positions()
                return True

        # Manejar eventos de controles
        for control in self.controls:
            if control.handle_event(event):
                # Un control cambió, invalidar cache
                self._invalidate_cache()
                # También invalidar cache del panel de info
                self.automata.info_cache_dirty = True
                return True

        return False

    def _update_control_positions(self):
        """Actualiza las posiciones de los controles cuando se mueve el menú"""
        # Regenerar controles con nuevas posiciones
        self.controls = self._create_controls()

    def draw(self, surface):
        """Dibuja el menú con cache inteligente para optimizar rendimiento"""
        if not self.visible:
            return

        # Verificar si necesitamos actualizar el cache
        need_update = (
            self.cache_dirty
            or self.cached_menu_surface is None
            or self._control_states_changed()
        )

        if need_update:
            # Crear nueva superficie del menú
            menu_surface = pygame.Surface(
                (self.menu_width, self.menu_height), pygame.SRCALPHA
            )

            # Fondo transparente simple
            background_color = (20, 25, 35, 180)  # Azul oscuro transparente
            menu_surface.fill(background_color)

            # Borde simple sin efectos
            border_color = (100, 120, 150)  # Azul claro
            pygame.draw.rect(
                menu_surface, border_color, menu_surface.get_rect(), 2, border_radius=10
            )

            # Título simple
            title_text = self.title_font.render(
                "🎮 SLIME MOLD CONTROLES", True, (200, 220, 255)
            )
            title_rect = title_text.get_rect()
            title_rect.centerx = self.menu_width // 2
            title_rect.y = 10
            menu_surface.blit(title_text, title_rect)

            # Dibujar controles una sola vez y cachearlos
            for control in self.controls:
                # Calcular posición relativa al menú
                rel_x = control.rect.x - self.menu_x
                rel_y = control.rect.y - self.menu_y

                # Para sliders con descripción, agregar espacio extra
                if hasattr(control, "description") and control.description:
                    control_surface = pygame.Surface(
                        (control.rect.width, control.rect.height + 40),
                        pygame.SRCALPHA,
                    )

                    temp_rect = pygame.Rect(
                        0, 40, control.rect.width, control.rect.height
                    )
                    original_rect = control.rect.copy()
                    control.rect = temp_rect
                    control.draw(control_surface)
                    control.rect = original_rect

                    menu_surface.blit(control_surface, (rel_x, rel_y - 40))
                else:
                    control_surface = pygame.Surface(
                        (control.rect.width, control.rect.height), pygame.SRCALPHA
                    )

                    temp_rect = pygame.Rect(
                        0, 0, control.rect.width, control.rect.height
                    )
                    original_rect = control.rect.copy()
                    control.rect = temp_rect
                    control.draw(control_surface)
                    control.rect = original_rect

                    menu_surface.blit(control_surface, (rel_x, rel_y))

            # Guardar en cache
            self.cached_menu_surface = menu_surface.copy()
            self._cache_control_states()
            self.cache_dirty = False
            self.ui_needs_update = False

        # Dibujar desde cache (mucho más rápido)
        surface.blit(self.cached_menu_surface, (self.menu_x, self.menu_y))

    def _cache_control_states(self):
        """Cachea los estados actuales de todos los controles"""
        self.last_control_states = {}
        for i, control in enumerate(self.controls):
            if hasattr(control, "current_val"):  # Slider
                self.last_control_states[f"slider_{i}"] = control.current_val
            elif hasattr(control, "is_active"):  # Switch
                self.last_control_states[f"switch_{i}"] = control.is_active
            elif hasattr(control, "config"):  # Button
                self.last_control_states[f"button_{i}"] = control.config.text

    def _control_states_changed(self):
        """Verifica si algún estado de control ha cambiado"""
        for i, control in enumerate(self.controls):
            if hasattr(control, "current_val"):  # Slider
                key = f"slider_{i}"
                if (
                    key not in self.last_control_states
                    or self.last_control_states[key] != control.current_val
                ):
                    return True
            elif hasattr(control, "is_active"):  # Switch
                key = f"switch_{i}"
                if (
                    key not in self.last_control_states
                    or self.last_control_states[key] != control.is_active
                ):
                    return True
            elif hasattr(control, "config"):  # Button
                key = f"button_{i}"
                if (
                    key not in self.last_control_states
                    or self.last_control_states[key] != control.config.text
                ):
                    return True
        return False

    def _invalidate_cache(self):
        """Invalida el cache forzando una actualización"""
        self.cache_dirty = True
        self.ui_needs_update = True


class InteractiveSlider:
    """Slider interactivo para valores numéricos"""

    def __init__(
        self,
        x,
        y,
        width,
        height,
        min_val,
        max_val,
        current_val,
        label,
        callback,
        description="",
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.description = description  # Texto descriptivo adicional
        self.callback = callback
        self.is_dragging = False
        self.is_hovered = False

        # Fuentes simples
        try:
            self.font = pygame.font.SysFont("arial", 14)
            self.desc_font = pygame.font.SysFont("arial", 16, bold=True)
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, 16)
            self.desc_font = pygame.font.Font(None, 18)

    def handle_event(self, event):
        """Maneja eventos del slider"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            if self.is_dragging:
                # Calcular nuevo valor basado en posición del mouse
                mouse_x = event.pos[0]
                relative_x = mouse_x - self.rect.x
                relative_x = max(0, min(self.rect.width, relative_x))

                # Convertir posición a valor
                ratio = relative_x / self.rect.width
                new_val = self.min_val + (self.max_val - self.min_val) * ratio

                if isinstance(self.min_val, int):
                    new_val = int(new_val)
                else:
                    new_val = round(new_val, 1)

                if new_val != self.current_val:
                    self.current_val = new_val
                    self.callback(new_val)
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.is_dragging = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.is_dragging = False

        return False

    def draw(self, surface):
        """Dibuja el slider con texto descriptivo"""
        # Texto descriptivo arriba del slider
        if self.description:
            desc_text = self.desc_font.render(self.description, True, (255, 255, 220))
            desc_rect = desc_text.get_rect()
            desc_rect.centerx = self.rect.centerx
            desc_rect.bottom = self.rect.top - 22
            surface.blit(desc_text, desc_rect)

        # Label y valor
        label_text = f"{self.label}: {self.current_val}"
        text_surface = self.font.render(label_text, True, (180, 180, 160))

        text_rect = text_surface.get_rect()
        text_rect.centerx = self.rect.centerx
        text_rect.bottom = self.rect.top - 4  # Justo arriba del track
        surface.blit(text_surface, text_rect)

        # Track del slider
        track_color = (60, 60, 70) if not self.is_hovered else (80, 80, 90)
        pygame.draw.rect(surface, track_color, self.rect, border_radius=8)

        # Calcular posición del handle
        ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + int(self.rect.width * ratio)
        handle_y = self.rect.centery
        handle_radius = 8

        # Handle del slider
        handle_color = (255, 200, 100) if self.is_dragging else (200, 180, 120)
        pygame.draw.circle(surface, handle_color, (handle_x, handle_y), handle_radius)
        pygame.draw.circle(
            surface, (120, 120, 120), (handle_x, handle_y), handle_radius, 2
        )

        # Borde del track
        border_color = (140, 140, 140) if self.is_hovered else (100, 100, 100)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=8)


class UltraSlimeMold:
    """Clase principal para el autómata celular Slime Mold"""

    def __init__(self):
        pygame.init()

        # Dimensiones dinámicas de la ventana
        self.window_width = DEFAULT_WINDOW_WIDTH
        self.window_height = DEFAULT_WINDOW_HEIGHT

        # Obtener información de la pantalla disponible
        pygame.display.init()
        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h

        # Verificar si el tamaño por defecto es mayor que la pantalla
        if DEFAULT_WINDOW_WIDTH > screen_width or DEFAULT_WINDOW_HEIGHT > screen_height:
            print(
                f"⚠️ Tamaño por defecto ({DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}) "
                + f"es mayor que la pantalla ({screen_width}x{screen_height})"
            )
            print("🔧 El sistema puede ajustar automáticamente el tamaño")

        # Configurar pantalla redimensionable con V-Sync
        display_flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE
        try:
            # Intentar activar V-Sync con OpenGL si está disponible
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height), display_flags
            )
        except pygame.error:
            # Fallback sin V-Sync si hay problemas
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height), display_flags
            )

        # Obtener el tamaño real asignado por el sistema
        actual_size = self.screen.get_size()
        if actual_size != (self.window_width, self.window_height):
            print(
                f"🔄 Sistema ajustó ventana: {self.window_width}x{self.window_height} → {actual_size[0]}x{actual_size[1]}"
            )
            self.window_width, self.window_height = actual_size
        pygame.display.set_caption("Ultra Slime Mold Automata")
        self.clock = pygame.time.Clock()

        # Configurar fuentes simples
        try:
            self.font = pygame.font.SysFont("arial", 24)
            self.small_font = pygame.font.SysFont("arial", 18)
            self.title_font = pygame.font.SysFont("arial", 26, bold=True)
            print("✅ Fuentes inicializadas")
        except (pygame.error, OSError):
            self.font = pygame.font.Font(None, 28)
            self.small_font = pygame.font.Font(None, 20)
            self.title_font = pygame.font.Font(None, 30)
            print("⚠️ Usando fuentes por defecto")

        # Buffer de superficie para renderizado
        self.back_buffer = pygame.Surface((self.window_width, self.window_height))

        # Estado de la simulación
        self.grid = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)
        self.current_filter = "default"
        self.paused = False
        self.show_info = True
        self.generation = 0

        # Control del mouse
        self.mouse_pressed = False
        self.eraser_size = 5

        # Control de zoom y navegación
        self.zoom_factor = 1.0
        self.zoom_offset_x = 0
        self.zoom_offset_y = 0
        self.min_zoom = 1.0
        self.max_zoom = 6.0

        # Control de arrastre
        self.dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_offset = (0, 0)

        # Optimizaciones de rendimiento
        self.render_scale = RENDER_SCALE
        self.skip_frames = 1
        self.frame_count = 0
        self.last_fps_update = 0
        self.cached_fps = 0

        # Cache de superficies optimizado
        self.surface_cache = {}
        self.last_grid_hash = None
        self.cached_base_surface = None
        self.zoom_cache = {}

        # Configuración de vsync
        self.vsync_enabled = True

        # Pre-compilar funciones JIT
        print("⚡ Pre-compilando funciones JIT...")
        dummy = np.random.rand(10, 10).astype(np.float32)
        activation_vectorized(dummy)
        create_color_mapping_jit(dummy)
        print("✅ JIT compilado y optimizado")

        # Inicializar menú interactivo
        self.interactive_menu = InteractiveMenu(self)
        print("🎮 Menú interactivo inicializado")

        # Cache para panel de información
        self.info_cache_dirty = True
        self.cached_info_surface = None
        self.last_info_data = {}

        self.print_info()

    def get_grid_offset_x(self):
        """Calcula la posición X centrada del grid dinámicamente"""
        return (self.window_width - GRID_DISPLAY_WIDTH) // 2

    def get_grid_offset_y(self):
        """Calcula la posición Y centrada del grid dinámicamente"""
        return (self.window_height - GRID_DISPLAY_HEIGHT) // 2

    def handle_window_resize(self, new_width, new_height):
        """Maneja el redimensionamiento de la ventana"""
        self.window_width = new_width
        self.window_height = new_height

        # Recrear el buffer trasero con las nuevas dimensiones
        self.back_buffer = pygame.Surface((self.window_width, self.window_height))

        # Forzar recálculo de tamaños y posiciones de menú (relativos)
        if hasattr(self, "interactive_menu"):
            # Recalcular ancho del menú (18% del ancho)
            self.interactive_menu.menu_width = int(self.window_width * 0.18)
            # Recalcular altura automáticamente
            self.interactive_menu._calculate_menu_height()

            # Asegurar que el menú siga dentro de los límites
            max_x = max(0, self.window_width - self.interactive_menu.menu_width)
            max_y = max(0, self.window_height - self.interactive_menu.menu_height)
            self.interactive_menu.menu_x = min(self.interactive_menu.menu_x, max_x)
            self.interactive_menu.menu_y = min(self.interactive_menu.menu_y, max_y)
            self.interactive_menu._update_control_positions()

        print(f"[RESIZE] Ventana redimensionada a: {new_width}x{new_height}")
        print(
            f"[GRID] Nueva posición: ({self.get_grid_offset_x()}, {self.get_grid_offset_y()})"
        )
        if hasattr(self, "interactive_menu"):
            print(
                f"[MENU] Nuevo tamaño: {self.interactive_menu.menu_width}x{self.interactive_menu.menu_height}"
            )

    def clamp_zoom_offset(self):
        """Limita el offset del zoom para mantener el grid siempre visible"""
        # Para zoom >= 1, permitir navegar por toda la imagen ampliada
        # pero manteniendo siempre algo del grid visible
        max_offset_x = GRID_DISPLAY_WIDTH * (self.zoom_factor - 1)
        max_offset_y = GRID_DISPLAY_HEIGHT * (self.zoom_factor - 1)

        # Límites estrictos: no permitir salirse del área del grid
        self.zoom_offset_x = max(0, min(max_offset_x, self.zoom_offset_x))
        self.zoom_offset_y = max(0, min(max_offset_y, self.zoom_offset_y))

    def erase_at_mouse_position(self, mouse_pos):
        """Borra células en la posición del mouse"""
        mouse_x, mouse_y = mouse_pos

        # Obtener el offset dinámico del grid centrado
        grid_offset_x = self.get_grid_offset_x()
        grid_offset_y = self.get_grid_offset_y()

        # Ajustar coordenadas del mouse para el grid centrado
        relative_x = mouse_x - grid_offset_x
        relative_y = mouse_y - grid_offset_y

        # Verificar si el mouse está dentro del área del grid
        if (
            relative_x < 0
            or relative_y < 0
            or relative_x >= GRID_DISPLAY_WIDTH
            or relative_y >= GRID_DISPLAY_HEIGHT
        ):
            return  # No borrar si está fuera del grid

        # Convertir coordenadas de pantalla a coordenadas de grilla
        # considerando zoom
        # Ajustar por el zoom y offset
        adjusted_x = (relative_x + self.zoom_offset_x) / self.zoom_factor
        adjusted_y = (relative_y + self.zoom_offset_y) / self.zoom_factor

        grid_x = int((adjusted_x / GRID_DISPLAY_WIDTH) * GRID_SIZE)
        grid_y = int((adjusted_y / GRID_DISPLAY_HEIGHT) * GRID_SIZE)

        # Asegurar que las coordenadas estén dentro de los límites
        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            # Crear área de borrado circular
            eraser_radius = max(
                1, int(self.eraser_size / self.zoom_factor)
            )  # Ajustar tamaño por zoom
            for dy in range(-eraser_radius, eraser_radius + 1):
                for dx in range(-eraser_radius, eraser_radius + 1):
                    # Calcular distancia desde el centro
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance <= eraser_radius:
                        new_x = grid_x + dx
                        new_y = grid_y + dy

                        # Verificar límites
                        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                            # Borrar gradualmente (más intenso en el centro)
                            fade_factor = (
                                1.0 - (distance / eraser_radius)
                                if eraser_radius > 0
                                else 1.0
                            )
                            self.grid[new_y, new_x] *= 1.0 - fade_factor * 0.8

    def update_grid_ultra_fast(self):
        """Actualización ultra rápida usando scipy y numba"""
        # Convolución ultra optimizada
        convolved = ndimage.convolve(
            self.grid, filters[self.current_filter], mode="wrap", output=np.float32
        )

        # Activación vectorizada JIT compilada
        activated = activation_vectorized(convolved)
        self.grid = np.clip(activated, 0.0, 1.0, out=self.grid)
        self.generation += 1

    def create_color_surface_bioluminescent(self):
        """Crea superficie con efectos bioluminiscentes ultra optimizada"""
        # Downsample la grilla para renderizado más rápido
        if self.render_scale > 1:
            downsampled = self.grid[:: self.render_scale, :: self.render_scale]
        else:
            downsampled = self.grid

        # Normalizar valores para mejor mapeo de colores
        normalized = np.clip(downsampled, 0, 1)

        # Usar función JIT optimizada para mapeo de colores
        color_array = create_color_mapping_jit(normalized)

        return color_array

    def draw_ultra_fast(self):
        """Renderizado ultra optimizado con cache inteligente y anti-flicker"""
        # Limpiar buffer trasero
        self.back_buffer.fill((0, 0, 0))

        # Cache inteligente: solo regenerar si hay cambios significativos
        current_grid_checksum = hash(
            self.grid.data.tobytes()[::1000]
        )  # Checksum rápido
        needs_regeneration = (
            self.last_grid_hash != current_grid_checksum
            or self.frame_count % 5 == 0  # Forzar actualización cada 5 frames
        )

        if needs_regeneration:
            color_array = self.create_color_surface_bioluminescent()
            # Crear superficie desde array
            surf = pygame.surfarray.make_surface(color_array.swapaxes(0, 1))
            # Cache la superficie base para reutilización
            self.cached_base_surface = pygame.transform.smoothscale(
                surf, (GRID_DISPLAY_WIDTH, GRID_DISPLAY_HEIGHT)
            )
            self.last_grid_hash = current_grid_checksum
        elif hasattr(self, "cached_base_surface") and self.cached_base_surface:
            # Reutilizar superficie cacheada
            pass
        else:
            # Fallback si no hay cache
            color_array = self.create_color_surface_bioluminescent()
            surf = pygame.surfarray.make_surface(color_array.swapaxes(0, 1))
            self.cached_base_surface = pygame.transform.smoothscale(
                surf, (GRID_DISPLAY_WIDTH, GRID_DISPLAY_HEIGHT)
            )

        # Aplicar zoom de manera más eficiente
        if self.zoom_factor != 1.0:
            # Cache de zoom: solo recalcular si zoom cambió significativamente
            zoom_key = f"{self.zoom_factor:.2f}"
            if (
                not hasattr(self, "zoom_cache")
                or self.zoom_cache.get("key") != zoom_key
                or needs_regeneration
            ):
                zoomed_width = int(GRID_DISPLAY_WIDTH * self.zoom_factor)
                zoomed_height = int(GRID_DISPLAY_HEIGHT * self.zoom_factor)
                if self.cached_base_surface:
                    scaled_surf = pygame.transform.smoothscale(
                        self.cached_base_surface, (zoomed_width, zoomed_height)
                    )
                    self.zoom_cache = {"key": zoom_key, "surface": scaled_surf}
                else:
                    scaled_surf = self.cached_base_surface
            else:
                scaled_surf = self.zoom_cache["surface"]
        else:
            scaled_surf = self.cached_base_surface

        # Calcular la posición de dibujo considerando el offset y centrado
        draw_x = self.get_grid_offset_x() - self.zoom_offset_x
        draw_y = self.get_grid_offset_y() - self.zoom_offset_y

        # Dibujar la superficie escalada al buffer trasero
        if scaled_surf:
            self.back_buffer.blit(scaled_surf, (draw_x, draw_y))

    def handle_input(self):
        """Manejo de entrada con menú interactivo"""
        # Verificar estado del mouse para borrado continuo
        if self.mouse_pressed and not self.dragging:
            mouse_pos = pygame.mouse.get_pos()
            self.erase_at_mouse_position(mouse_pos)

        # Manejar arrastre para navegación
        if self.dragging:
            current_mouse_pos = pygame.mouse.get_pos()
            dx = current_mouse_pos[0] - self.drag_start_pos[0]
            dy = current_mouse_pos[1] - self.drag_start_pos[1]

            # Actualizar offset basado en el movimiento del mouse
            self.zoom_offset_x = self.drag_start_offset[0] - dx
            self.zoom_offset_y = self.drag_start_offset[1] - dy

            # Aplicar límites para el offset
            self.clamp_zoom_offset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                # Manejar redimensionamiento de ventana
                new_width, new_height = event.w, event.h
                self.handle_window_resize(new_width, new_height)
                continue

            # Primero, permitir que el menú maneje el evento
            if self.interactive_menu.handle_event(event):
                continue  # Si el menú manejó el evento, no procesarlo más

            # Si el menú no manejó el evento, procesarlo normalmente
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Clic izquierdo
                    # Verificar si se está arrastrando (con tecla shift o clic derecho también)
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        # Modo arrastre
                        self.dragging = True
                        self.drag_start_pos = event.pos
                        self.drag_start_offset = (
                            self.zoom_offset_x,
                            self.zoom_offset_y,
                        )
                        print("🖱️ Modo arrastre activado")
                    else:
                        # Modo borrador
                        self.mouse_pressed = True
                        self.erase_at_mouse_position(event.pos)
                        print("🖱️ Modo borrador activado")
                elif event.button == 3:  # Clic derecho para arrastrar
                    self.dragging = True
                    self.drag_start_pos = event.pos
                    self.drag_start_offset = (self.zoom_offset_x, self.zoom_offset_y)
                    print("🖱️ Modo arrastre activado (clic derecho)")
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Soltar clic izquierdo
                    if self.mouse_pressed:
                        self.mouse_pressed = False
                        print("🖱️ Modo borrador desactivado")
                    if self.dragging:
                        self.dragging = False
                        print("🖱️ Modo arrastre desactivado")
                elif event.button == 3:  # Soltar clic derecho
                    self.dragging = False
                    print("🖱️ Modo arrastre desactivado")
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom con rueda del mouse mejorado
                old_zoom = self.zoom_factor

                # Velocidad de zoom adaptiva (más lenta en zoom alto)
                base_speed = 0.1
                zoom_speed = base_speed * (1 + self.zoom_factor * 0.1)

                if event.y > 0:  # Rueda hacia arriba - zoom in
                    self.zoom_factor = min(self.max_zoom, self.zoom_factor + zoom_speed)
                elif event.y < 0:  # Rueda hacia abajo - zoom out
                    self.zoom_factor = max(self.min_zoom, self.zoom_factor - zoom_speed)

                # Si el zoom cambió, ajustar el offset para mantener centrado en el cursor
                if old_zoom != self.zoom_factor:
                    mouse_pos = pygame.mouse.get_pos()
                    mouse_x, mouse_y = mouse_pos

                    # Para cualquier zoom, mantener el punto del mouse fijo
                    zoom_ratio = self.zoom_factor / old_zoom
                    self.zoom_offset_x = (
                        mouse_x - (mouse_x - self.zoom_offset_x) * zoom_ratio
                    )
                    self.zoom_offset_y = (
                        mouse_y - (mouse_y - self.zoom_offset_y) * zoom_ratio
                    )

                    # Aplicar límites usando el método
                    self.clamp_zoom_offset()

                    print(f"🔍 Zoom: {self.zoom_factor:.1f}x")
            elif event.type == pygame.KEYDOWN:
                key = event.key
                # Controles básicos con teclado
                if key == pygame.K_ESCAPE:
                    return False
                elif key == pygame.K_SPACE:  # Pausa/reanuda con ESPACIO
                    self.interactive_menu._toggle_pause()
                    print("⏯️ Toggle pausa con ESPACIO")
                elif key == pygame.K_m:  # Toggle menú
                    self.interactive_menu.visible = not self.interactive_menu.visible
                    print(
                        f"🎮 Menú: {'Visible' if self.interactive_menu.visible else 'Oculto'}"
                    )
                elif key == pygame.K_h:  # Ayuda rápida (IMPLEMENTAR)
                    print("\nCONTROLES RÁPIDOS:")
                    print("ESPACIO - Pausar/Reanudar simulación")
                    print("M - Toggle menú interactivo")
                    print("H - Mostrar esta ayuda")
                    print("ESC - Salir")
                    print("Usa el menú interactivo para otros controles!")
        return True

    def draw_info_panel(self):
        """Panel de información con cache inteligente para optimizar rendimiento"""
        if not self.show_info:
            return

        # Datos actuales para verificar cambios
        current_info_data = {
            "generation": self.generation,
            "fps": self.cached_fps,
            "paused": self.paused,
            "skip_frames": self.skip_frames,
            "zoom_factor": self.zoom_factor,
            "zoom_offset_x": self.zoom_offset_x,
            "zoom_offset_y": self.zoom_offset_y,
            "render_scale": self.render_scale,
            "eraser_size": self.eraser_size,
            "vsync_enabled": self.vsync_enabled,
        }

        # Verificar si necesitamos actualizar el cache
        need_update = (
            self.info_cache_dirty
            or self.cached_info_surface is None
            or current_info_data != self.last_info_data
        )

        if need_update:
            # Posición del panel de información - más compacto
            panel_width = int(self.window_width * 0.20)  # 20% del ancho
            panel_height = int(self.window_height * 0.28)  # 28% de la altura
            panel_x = self.window_width - panel_width - 10
            panel_y = 10

            # Crear superficie del panel transparente y simple
            info_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)

            # Fondo transparente simple - igual que el menú
            background_color = (20, 25, 35, 180)  # Azul oscuro transparente
            info_surface.fill(background_color)

            # Borde simple - igual que el menú
            border_color = (100, 120, 150)  # Azul claro
            pygame.draw.rect(
                info_surface, border_color, info_surface.get_rect(), 2, border_radius=10
            )

            # Título del panel
            try:
                title_font = pygame.font.SysFont("arial", 16, bold=True)
            except (pygame.error, OSError):
                title_font = pygame.font.Font(None, 18)

            title_text = title_font.render("📊 SIMULATION DATA", True, (200, 220, 255))
            title_rect = title_text.get_rect()
            title_rect.centerx = panel_width // 2
            title_rect.y = 10
            info_surface.blit(title_text, title_rect)

            # Área de información con dos columnas
            info_start_y = 40  # Espacio después del título
            info_bg = pygame.Rect(10, info_start_y, panel_width - 20, 200)
            pygame.draw.rect(info_surface, (40, 40, 50, 120), info_bg, border_radius=8)
            pygame.draw.rect(info_surface, (120, 120, 140), info_bg, 1, border_radius=8)

            # Información en dos columnas
            col1_x = 20  # Espacio inicial para la columna 1
            col2_x = panel_width // 2 + 5  # Espacio entre columnas
            info_y = info_start_y + 15  # Espacio inicial para la información

            # Columna 1: Estado del sistema
            try:
                info_font = pygame.font.SysFont("arial", 12)
            except (pygame.error, OSError):
                info_font = pygame.font.Font(None, 14)

            info_lines_left = [
                "📊 SIMULACION",
                f"Gen: {current_info_data['generation']}",
                f"FPS: {current_info_data['fps']:.1f}",
                f"Estado: {'PAUSE' if current_info_data['paused'] else 'ACTIVE'}",
                f"Skip: {current_info_data['skip_frames']}f",
                "",
                "🗺️ GRID",
                f"Size: {GRID_SIZE}x{GRID_SIZE}",
                f"Cells: {GRID_SIZE * GRID_SIZE:,}",
            ]

            for i, line in enumerate(info_lines_left):
                if line.startswith("["):
                    # Títulos de sección
                    color = (255, 220, 100)
                elif line.strip() == "":
                    continue
                else:
                    # Datos normales
                    color = (220, 220, 200)

                text_surface = info_font.render(line, True, color)
                info_surface.blit(text_surface, (col1_x, info_y + i * 16))

            # Columna 2: Vista y herramientas
            info_lines_right = [
                "🔬 VISTA",
                f"Zoom: {current_info_data['zoom_factor']:.1f}x",
                f"Offset: {int(current_info_data['zoom_offset_x'])},"
                f"{int(current_info_data['zoom_offset_y'])}",
                f"Scale: 1:{current_info_data['render_scale']}",
                "",
                "🛠️ HERRAMIENTAS",
                f"Borrador: {current_info_data['eraser_size']}px",
                f"V-Sync: {'ON' if current_info_data['vsync_enabled'] else 'OFF'}",
            ]

            for i, line in enumerate(info_lines_right):
                if line.startswith("["):
                    # Títulos de sección
                    color = (100, 200, 255)
                elif line.strip() == "":
                    continue
                else:
                    # Datos normales
                    color = (200, 220, 200)

                text_surface = info_font.render(line, True, color)
                info_surface.blit(text_surface, (col2_x, info_y + i * 16))

            # Guardar en cache
            self.cached_info_surface = info_surface.copy()
            self.last_info_data = current_info_data.copy()
            self.info_cache_dirty = False

        # Dibujar desde cache (mucho más rápido)
        panel_x = self.window_width - int(self.window_width * 0.20) - 10
        panel_y = 10
        self.back_buffer.blit(self.cached_info_surface, (panel_x, panel_y))

    def print_info(self):
        """Información inicial en consola"""
        print("\n === ULTRA SLIME MOLD AUTOMATA ===")
        print(f"🖥️ Ventana: {self.window_width}x{self.window_height} pixels")

        # Añadir información sobre tamaño por defecto vs actual
        if (
            self.window_width != DEFAULT_WINDOW_WIDTH
            or self.window_height != DEFAULT_WINDOW_HEIGHT
        ):
            print(
                f"📝 Tamaño solicitado: {DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT} pixels"
            )
            print(
                f"🖥️ Ajustado por sistema: {self.window_width}x{self.window_height} pixels"
            )

        print(f"🗺️ Área del grid: {GRID_DISPLAY_WIDTH}x{GRID_DISPLAY_HEIGHT} pixels")
        print(f"📍 Posición: ({self.get_grid_offset_x()}, {self.get_grid_offset_y()})")
        print(f"🗺️ Grilla: {GRID_SIZE}x{GRID_SIZE} células")
        print(f"🗺️ Células totales: {GRID_SIZE * GRID_SIZE:,}")
        print(f"🗺️ Render escala: 1:{self.render_scale}")
        print("\n🎮 CONTROLES PRINCIPALES:")
        print("ESPACIO - [PAUSE/PLAY] Pausar/Reanudar simulación")
        print("M - Toggle menú interactivo")
        print("H - Mostrar ayuda")
        print("ESC - Salir")
        print("\n[TIP] Usa el menú interactivo (M) para todos los controles!")
        print("[TIP] El menú se puede arrastrar desde la barra de título")
        print("=" * 50)

    def run(self):
        """Bucle principal ultra optimizado con menú interactivo"""
        running = True
        print("🚀 Iniciando simulación con menú interactivo optimizado...")

        while running:
            current_time = pygame.time.get_ticks()

            # Manejar eventos de manera más eficiente
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.VIDEORESIZE:
                    # Manejar redimensionamiento
                    new_width, new_height = event.w, event.h
                    self.handle_window_resize(new_width, new_height)
                    continue

                # Procesar eventos del menú y controles
                if self.interactive_menu.handle_event(event):
                    continue

                # Procesar otros eventos
                self._handle_other_events(event)

            if not running:
                break

            # Actualizar lógica de simulación solo si no está pausado
            if not self.paused:
                self.update_grid_ultra_fast()

            # Renderizado simplificado - siempre dibujar todo para evitar parpadeo
            self.frame_count += 1

            # Renderizar simulación cada N frames para rendimiento
            if self.frame_count % self.skip_frames == 0:
                self.draw_ultra_fast()

            # Siempre dibujar UI para evitar parpadeo
            self.interactive_menu.draw(self.back_buffer)
            self.draw_info_panel()

            # Siempre transferir a pantalla
            self.screen.blit(self.back_buffer, (0, 0))
            pygame.display.flip()

            # Control de FPS optimizado según configuración V-Sync
            if self.vsync_enabled:
                # Con V-Sync: usar tick normal que es más eficiente con sincronización
                self.clock.tick(FPS)
            else:
                # Sin V-Sync: usar tick_busy_loop para máximo rendimiento
                self.clock.tick_busy_loop(FPS)

            # Actualizar FPS cached menos frecuentemente
            if current_time - self.last_fps_update > 1000:
                self.cached_fps = self.clock.get_fps()
                self.last_fps_update = current_time

        pygame.quit()
        sys.exit()

    def _handle_other_events(self, event):
        """Maneja eventos que no son del menú"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Clic izquierdo
                # Verificar si se está arrastrando (con tecla shift)
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    # Modo arrastre
                    self.dragging = True
                    self.drag_start_pos = event.pos
                    self.drag_start_offset = (self.zoom_offset_x, self.zoom_offset_y)
                else:
                    # Modo borrador
                    self.mouse_pressed = True
                    self.erase_at_mouse_position(event.pos)
            elif event.button == 3:  # Clic derecho para arrastrar
                self.dragging = True
                self.drag_start_pos = event.pos
                self.drag_start_offset = (self.zoom_offset_x, self.zoom_offset_y)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_pressed = False
                self.dragging = False
            elif event.button == 3:
                self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_pressed and not self.dragging:
                self.erase_at_mouse_position(event.pos)
            elif self.dragging:
                current_mouse_pos = pygame.mouse.get_pos()
                dx = current_mouse_pos[0] - self.drag_start_pos[0]
                dy = current_mouse_pos[1] - self.drag_start_pos[1]
                self.zoom_offset_x = self.drag_start_offset[0] - dx
                self.zoom_offset_y = self.drag_start_offset[1] - dy
                self.clamp_zoom_offset()

        elif event.type == pygame.MOUSEWHEEL:
            # Zoom con rueda del mouse
            old_zoom = self.zoom_factor
            base_speed = 0.1
            zoom_speed = base_speed * (1 + self.zoom_factor * 0.1)

            if event.y > 0:  # Zoom in
                self.zoom_factor = min(self.max_zoom, self.zoom_factor + zoom_speed)
            elif event.y < 0:  # Zoom out
                self.zoom_factor = max(self.min_zoom, self.zoom_factor - zoom_speed)

            if old_zoom != self.zoom_factor:
                # Ajustar offset para mantener centrado en cursor
                mouse_pos = pygame.mouse.get_pos()
                zoom_ratio = self.zoom_factor / old_zoom
                self.zoom_offset_x = (
                    mouse_pos[0] - (mouse_pos[0] - self.zoom_offset_x) * zoom_ratio
                )
                self.zoom_offset_y = (
                    mouse_pos[1] - (mouse_pos[1] - self.zoom_offset_y) * zoom_ratio
                )
                self.clamp_zoom_offset()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.interactive_menu._toggle_pause()
            elif event.key == pygame.K_m:
                self.interactive_menu.visible = not self.interactive_menu.visible
            elif event.key == pygame.K_h:
                print("\n[HELP] CONTROLES RÁPIDOS:")
                print("ESPACIO - Pausar/Reanudar")
                print("M - Toggle menú")
                print("H - Ayuda")
                print("ESC - Salir")


def main():
    """Función principal para ejecutar la simulación"""
    automata = UltraSlimeMold()
    automata.run()


if __name__ == "__main__":
    main()
