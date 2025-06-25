"""Menú interactivo principal para el autómata"""

import numpy as np
import pygame
from scipy import ndimage

from app.config.constants import (
    HEADER_HEIGHT,
    MAX_ERASER_SIZE,
    MAX_ZOOM,
    MENU_WIDTH_RATIO,
    MIN_ERASER_SIZE,
    MIN_ZOOM,
)
from app.utils.math_utils import create_circular_kernel

from .button import InteractiveButton
from .button_config import ButtonConfig
from .slider import InteractiveSlider
from .switch import InteractiveSwitch


class InteractiveMenu:
    """Menú interactivo con controles optimizados y cache inteligente"""

    def __init__(self, automata_instance):
        self.automata = automata_instance
        self.visible = True
        self.dragging = False
        self.drag_offset = (0, 0)

        # Posición y tamaño del menú
        self.menu_x = 10
        self.menu_y = 10
        # Ancho del menú basado en el ancho de la ventana
        self.menu_width = int(self.automata.window_width * MENU_WIDTH_RATIO)

        # Calcular altura automáticamente basada en componentes
        self.header_height = HEADER_HEIGHT
        self._calculate_menu_height()

        # Sistema de cache inteligente optimizado
        self.cache_dirty = True
        self.cached_menu_surface = None
        self.last_control_states = {}
        self.ui_needs_update = True

        # Cache de valores para compatibilidad
        self.last_cached_values = {"zoom": 0, "eraser": 0, "skip": 0}

        # Configurar fuentes
        try:
            self.title_font = pygame.font.SysFont("arial", 18, bold=True)
            self.subtitle_font = pygame.font.SysFont("arial", 14)
        except (pygame.error, OSError):
            self.title_font = pygame.font.Font(None, 22)
            self.subtitle_font = pygame.font.Font(None, 18)

        # Efectos especiales usando kernels circulares
        self.special_effects = {"blur": False, "glow": False, "ripple": False}

        # Controles del menú
        self.controls = self._create_controls()

    def _calculate_menu_height(self):
        """Calcula la altura del menú automáticamente según componentes"""
        # Componentes del menú con sus respectivas alturas
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

        # Configuración de layout
        control_width = int(self.menu_width * 0.75)
        control_height = 30
        control_spacing = 35
        section_spacing = 15

        # Punto de inicio después del header
        start_y = self.menu_y + self.header_height + 15
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
        current_y = start_y + control_spacing * 7 + section_spacing * 2 + 5
        slider_width = int(self.menu_width * 0.8)
        slider_height = 20
        slider_spacing = 65

        # Crear sliders con descripciones
        sliders = [
            InteractiveSlider(
                col_x + (control_width - slider_width) // 2,
                current_y,
                slider_width,
                slider_height,
                MIN_ZOOM,
                MAX_ZOOM,
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
                MIN_ERASER_SIZE,
                MAX_ERASER_SIZE,
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
                1,  # ✅ min_val
                10,  # ✅ max_val
                self.automata.skip_frames,
                "⚡ Frame Skip",
                self._set_frame_skip,
                "⚡ SALTO DE FRAMES",
            ),
        ]

        # SECCIÓN 4: Botones de calidad
        current_y += slider_spacing * 3 + 20
        mini_width = int(self.menu_width * 0.38)
        mini_height = int(control_height * 0.9)
        mini_spacing = 8

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

    # Métodos de acción para los controles
    def _reset_grid(self):
        """Reinicia el grid con valores aleatorios"""
        import numpy as np

        from app.config.constants import GRID_SIZE

        self.automata.grid = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)
        self.automata.generation = 0
        print("🔄 Grid reiniciado desde menú")

    def _toggle_pause(self):
        """Pausa/reanuda la simulación"""
        self.automata.paused = not self.automata.paused
        # Actualizar texto del botón
        for control in self.controls:
            if hasattr(control, "config") and (
                "Pausar" in control.config.text or "Reanudar" in control.config.text
            ):
                if self.automata.paused:
                    control.config.text = "▶️ Reanudar"
                else:
                    control.config.text = "⏸️ Pausar"
        status = "Pausado" if self.automata.paused else "Reanudado"
        print(f"{status} desde menú")

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
        """Toggle V-Sync"""
        self.automata.vsync_enabled = not self.automata.vsync_enabled
        status = "ON" if self.automata.vsync_enabled else "OFF"
        print(f"📺 V-Sync: {status}")

    def _set_zoom(self, value):
        """Establece el zoom usando el slider"""
        rounded_value = round(value, 1)
        if abs(rounded_value - self.last_cached_values["zoom"]) > 0.2:
            self.automata.zoom_factor = value
            self.automata.clamp_zoom_offset()
            self.last_cached_values["zoom"] = rounded_value
            print(f"🔍 [ZOOM] {value:.1f}x")

    def _set_eraser_size(self, value):
        """Establece el tamaño del borrador usando el slider"""
        int_value = int(value)
        if int_value != self.last_cached_values["eraser"]:
            self.automata.eraser_size = int_value
            self.last_cached_values["eraser"] = int_value
            print(f"🖌️ [BRUSH] {int_value}px")

    def _set_frame_skip(self, value):
        """Establece el frame skip usando el slider"""
        int_value = int(value)
        if int_value != self.last_cached_values["skip"]:
            self.automata.skip_frames = int_value
            self.last_cached_values["skip"] = int_value
            print(f"⚡ [SKIP] {int_value} frames")

    def handle_event(self, event):
        """Maneja eventos del menú"""
        if not self.visible:
            return False

        # Manejar eventos de controles
        for control in self.controls:
            if control.handle_event(event):
                self._invalidate_cache()
                return True

        return False

    def draw(self, surface):
        """Dibuja el menú con cache inteligente"""
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

            # Fondo transparente
            background_color = (20, 25, 35, 180)
            menu_surface.fill(background_color)

            # Dibujar header
            header_rect = pygame.Rect(0, 0, self.menu_width, self.header_height)
            pygame.draw.rect(menu_surface, (30, 35, 45), header_rect)

            # Título del menú
            title_text = self.title_font.render("🎮 CONTROLES", True, (240, 240, 255))
            title_rect = title_text.get_rect(center=(self.menu_width // 2, 20))
            menu_surface.blit(title_text, title_rect)

            # Dibujar controles
            for control in self.controls:
                # Ajustar posición relativa al menú
                adjusted_surface = pygame.Surface(control.rect.size, pygame.SRCALPHA)
                if hasattr(control, "draw"):
                    # Crear una superficie temporal para el control
                    temp_rect = pygame.Rect(
                        0, 0, control.rect.width, control.rect.height
                    )
                    original_rect = control.rect
                    control.rect = temp_rect
                    control.draw(adjusted_surface)
                    control.rect = original_rect

                    # Blitear en posición relativa
                    menu_surface.blit(
                        adjusted_surface,
                        (control.rect.x - self.menu_x, control.rect.y - self.menu_y),
                    )

            # Actualizar cache
            self.cached_menu_surface = menu_surface
            self.cache_dirty = False
            self._cache_control_states()

        # Dibujar desde cache
        surface.blit(self.cached_menu_surface, (self.menu_x, self.menu_y))

    def _cache_control_states(self):
        """Cachea los estados actuales de todos los controles"""
        self.last_control_states = {}
        for i, control in enumerate(self.controls):
            if hasattr(control, "current_val"):  # Slider
                self.last_control_states[f"slider_{i}"] = control.current_val
            elif hasattr(control, "state"):  # Switch
                self.last_control_states[f"switch_{i}"] = control.state
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
            elif hasattr(control, "state"):  # Switch
                key = f"switch_{i}"
                if (
                    key not in self.last_control_states
                    or self.last_control_states[key] != control.state
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

    def apply_circular_effect(self, effect_type, intensity=1.0):
        """
        Aplica efectos especiales usando kernels circulares

        Args:
            effect_type: Tipo de efecto ('blur', 'glow', 'ripple')
            intensity: Intensidad del efecto (0.0 a 1.0)
        """
        if effect_type not in self.special_effects:
            return

        # Crear kernel circular para el efecto
        kernel_size = max(3, int(intensity * 10))
        kernel = create_circular_kernel(kernel_size)

        if effect_type == "blur":
            # Aplicar desenfoque circular
            self.automata.grid = ndimage.convolve(
                self.automata.grid, kernel, mode="constant"
            )
        elif effect_type == "glow":
            # Crear efecto de brillo
            glowed = ndimage.convolve(self.automata.grid, kernel * 0.3, mode="constant")
            self.automata.grid = np.maximum(self.automata.grid, glowed)
        elif effect_type == "ripple":
            # Efecto de ondas usando kernel circular
            rippled = ndimage.convolve(self.automata.grid, kernel * 0.1, mode="wrap")
            self.automata.grid = (self.automata.grid + rippled) * 0.5

        # Marcar como activado
        self.special_effects[effect_type] = True
