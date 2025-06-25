"""Clase principal del autómata celular Slime Mold"""

import numpy as np
import pygame
from scipy import ndimage

from app.config.constants import (
    DEFAULT_ERASER_SIZE,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    FPS,
    GRID_DISPLAY_HEIGHT,
    GRID_DISPLAY_WIDTH,
    GRID_SIZE,
    MAX_ZOOM,
    MIN_ZOOM,
    RENDER_SCALE,
)
from app.config.settings import SimulationSettings
from app.core.algorithms import (
    activation_vectorized,
    enhanced_color_mapping_with_smooth_transitions,
)
from app.rendering.colors import ColorMapper
from app.ui.components.menu import InteractiveMenu
from app.utils.math_utils import (
    angle_between_points,
    distance_2d,
    normalize_array,
    smooth_step,
)


class UltraSlimeMold:
    """Clase principal para el autómata celular Slime Mold"""

    def __init__(self, screen=None):
        if screen is None:
            pygame.init()
            # Configurar pantalla redimensionable con V-Sync
            display_flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.RESIZABLE
            try:
                screen = pygame.display.set_mode(
                    (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT), display_flags
                )
            except pygame.error:
                # Fallback sin V-Sync si hay problemas
                screen = pygame.display.set_mode(
                    (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT), display_flags
                )

        self.screen = screen
        self.window_width, self.window_height = screen.get_size()

        # Configurar fuentes
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

        # Clock para control de FPS
        self.clock = pygame.time.Clock()

        # Buffer de superficie para renderizado
        self.back_buffer = pygame.Surface((self.window_width, self.window_height))

        # Configuraciones
        self.settings = SimulationSettings()
        self.color_mapper = ColorMapper()

        # Estado de la simulación
        self.grid = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)
        self.current_filter = "default"
        self.paused = False
        self.show_info = self.settings.show_info
        self.generation = 0

        # Control del mouse
        self.mouse_pressed = False
        self.eraser_size = DEFAULT_ERASER_SIZE  # self.settings.initial_eraser_size

        # Control de zoom y navegación
        self.zoom_factor = self.settings.initial_zoom
        self.zoom_offset_x = 0
        self.zoom_offset_y = 0
        self.min_zoom = MIN_ZOOM
        self.max_zoom = MAX_ZOOM

        # Control de arrastre
        self.dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_offset = (0, 0)

        # Optimizaciones de rendimiento
        self.render_scale = RENDER_SCALE
        self.skip_frames = self.settings.initial_skip_frames
        self.frame_count = 0
        self.last_fps_update = 0
        self.cached_fps = 0

        # Cache de superficies optimizado
        self.surface_cache = {}
        self.last_grid_hash = None
        self.cached_base_surface = None
        self.zoom_cache = {}

        # Configuración de vsync
        self.vsync_enabled = self.settings.vsync_enabled

        # Cache para panel de información
        self.info_cache_dirty = True
        self.cached_info_surface = None
        self.last_info_data = {}

        # Inicializar menú interactivo
        self.interactive_menu = InteractiveMenu(self)
        print("🎮 Menú interactivo inicializado")

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
            # Recalcular ancho del menú (20% del ancho)
            self.interactive_menu.menu_width = int(self.window_width * 0.20)
            # Recalcular altura automáticamente
            self.interactive_menu._calculate_menu_height()

            # Asegurar que el menú siga dentro de los límites
            max_x = max(0, self.window_width - self.interactive_menu.menu_width)
            max_y = max(0, self.window_height - self.interactive_menu.menu_height)
            self.interactive_menu.menu_x = min(self.interactive_menu.menu_x, max_x)
            self.interactive_menu.menu_y = min(self.interactive_menu.menu_y, max_y)
            self.interactive_menu.controls = self.interactive_menu._create_controls()

        print(f"[RESIZE] Ventana redimensionada a: {new_width}x{new_height}")
        print(
            f"[GRID] Nueva posición: ({self.get_grid_offset_x()}, {self.get_grid_offset_y()})"
        )
        if hasattr(self, "interactive_menu"):
            print(f"[MENU] Menú redimensionado: {self.interactive_menu.menu_width}")

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
        """Borra células en la posición del mouse con borrador circular mejorado"""

        mouse_x, mouse_y = mouse_pos

        grid_offset_x = self.get_grid_offset_x()
        grid_offset_y = self.get_grid_offset_y()

        # Coordenadas relativas al grid
        relative_x = mouse_x - grid_offset_x
        relative_y = mouse_y - grid_offset_y

        # Verificar si está dentro del área del grid
        if (
            relative_x < 0
            or relative_y < 0
            or relative_x >= GRID_DISPLAY_WIDTH
            or relative_y >= GRID_DISPLAY_HEIGHT
        ):
            return False

        # Ajustar por zoom y offset
        adjusted_x = (relative_x + self.zoom_offset_x) / self.zoom_factor
        adjusted_y = (relative_y + self.zoom_offset_y) / self.zoom_factor

        # Convertir a coordenadas del grid
        grid_x = int((adjusted_x / GRID_DISPLAY_WIDTH) * GRID_SIZE)
        grid_y = int((adjusted_y / GRID_DISPLAY_HEIGHT) * GRID_SIZE)

        # Verificar límites del grid
        if not (0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE):
            return False

        # Aplicar borrador circular usando distance_2d para mayor precisión
        eraser_radius = max(1, int(self.eraser_size / self.zoom_factor))
        center_point = (grid_x, grid_y)

        # Borrado gradual: más intenso en el centro
        for dy in range(-eraser_radius, eraser_radius + 1):
            for dx in range(-eraser_radius, eraser_radius + 1):
                target_point = (grid_x + dx, grid_y + dy)
                distance = distance_2d(center_point, target_point)

                # Solo borrar si está dentro del radio del borrador
                if distance <= eraser_radius:
                    new_x, new_y = target_point

                    # Verificar límites
                    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                        # Borrado gradual (más intenso en el centro)
                        fade_factor = (
                            1.0 - (distance / eraser_radius)
                            if eraser_radius > 0
                            else 1.0
                        )
                        self.grid[new_y, new_x] *= 1.0 - fade_factor * 0.8

        return True

    def draw_eraser_preview(self, surface, mouse_pos):
        """Dibuja una preview del área del borrador con estados dinámicos"""
        # No mostrar si se está borrando activamente
        if self.mouse_pressed:
            return

        # No mostrar si se está arrastrando
        if self.dragging:
            return

        # Verificar si el mouse está sobre el área del grid
        grid_offset_x = self.get_grid_offset_x()
        grid_offset_y = self.get_grid_offset_y()

        relative_x = mouse_pos[0] - grid_offset_x
        relative_y = mouse_pos[1] - grid_offset_y

        # Solo mostrar preview si está dentro del grid
        if (
            relative_x < 0
            or relative_y < 0
            or relative_x >= GRID_DISPLAY_WIDTH
            or relative_y >= GRID_DISPLAY_HEIGHT
        ):
            return

        # Verificar si hay teclas modificadoras presionadas (shift para arrastrar)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            return  # No mostrar en modo arrastre

        # CORRECCIÓN: Usar el mismo cálculo que erase_at_mouse_position()
        # Calcular el radio del borrador en coordenadas del grid
        eraser_radius_grid = max(1, int(self.eraser_size / self.zoom_factor))

        # Convertir el radio del grid a píxeles de pantalla
        # Esto debe coincidir exactamente con el área de borrado
        eraser_radius_pixels = int(
            eraser_radius_grid * self.zoom_factor * (GRID_DISPLAY_WIDTH / GRID_SIZE)
        )

        # Asegurar límites razonables
        eraser_radius_pixels = max(2, min(eraser_radius_pixels, 200))

        # Colores para el preview
        fill_color = (200, 200, 255, 60)  # Azul suave semi-transparente para relleno
        border_color = (255, 255, 255, 120)  # Blanco más opaco para el borde

        # Crear superficie temporal para transparencia
        try:
            temp_surface = pygame.Surface(
                (eraser_radius_pixels * 2 + 4, eraser_radius_pixels * 2 + 4),
                pygame.SRCALPHA,
            )

            center_pos = (eraser_radius_pixels + 2, eraser_radius_pixels + 2)

            # Dibujar círculo relleno (fondo semi-transparente)
            pygame.draw.circle(
                temp_surface,
                fill_color,
                center_pos,
                eraser_radius_pixels,
                0,  # 0 = círculo relleno
            )

            # Dibujar borde del círculo (más visible)
            pygame.draw.circle(
                temp_surface,
                border_color,
                center_pos,
                eraser_radius_pixels,
                2,  # Grosor del borde
            )

            surface.blit(
                temp_surface,
                (
                    mouse_pos[0] - eraser_radius_pixels - 2,
                    mouse_pos[1] - eraser_radius_pixels - 2,
                ),
            )
        except pygame.error:
            # Fallback sin transparencia
            pygame.draw.circle(
                surface,
                (100, 100, 200),
                mouse_pos,
                eraser_radius_pixels,
                0,  # Círculo relleno
            )
            pygame.draw.circle(
                surface, (255, 255, 255), mouse_pos, eraser_radius_pixels, 2  # Borde
            )

    def apply_directional_erasing(self, start_pos, end_pos):
        """
        Aplica borrado direccional usando angle_between_points

        Args:
            start_pos: Posición inicial del borrado
            end_pos: Posición final del borrado
        """

        # Calcular ángulo y distancia del movimiento
        angle = angle_between_points(start_pos, end_pos)
        distance = distance_2d(start_pos, end_pos)

        # Solo aplicar borrado direccional si hay movimiento significativo
        if distance < 5:
            return

        # Convertir a coordenadas del grid
        grid_offset_x = self.get_grid_offset_x()
        grid_offset_y = self.get_grid_offset_y()

        # Aplicar borrado a lo largo de la línea
        steps = int(distance)
        for i in range(steps):
            # Interpolar posición a lo largo de la línea
            t = i / max(1, steps - 1)
            current_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            current_y = start_pos[1] + t * (end_pos[1] - start_pos[1])

            # Aplicar borrado en esta posición
            self.erase_at_mouse_position((int(current_x), int(current_y)))

    def apply_brush_stroke(self, stroke_points):
        """
        Aplica trazo de pincel usando múltiples puntos y ángulos

        Args:
            stroke_points: Lista de puntos del trazo [(x, y), ...]
        """

        if len(stroke_points) < 2:
            return

        for i in range(len(stroke_points) - 1):
            start_point = stroke_points[i]
            end_point = stroke_points[i + 1]

            # Calcular ángulo del trazo
            stroke_angle = angle_between_points(start_point, end_point)

            # Usar ángulo para determinar intensidad del borrado
            # Trazos más verticales son más intensos
            angle_factor = abs(np.sin(stroke_angle))
            intensity = smooth_step(0.0, 1.0, angle_factor)

            # Aplicar borrado con intensidad variable
            temp_eraser_size = self.eraser_size
            self.eraser_size = int(self.eraser_size * (0.5 + intensity * 0.5))

            self.apply_directional_erasing(start_point, end_point)

            # Restaurar tamaño original
            self.eraser_size = temp_eraser_size

    def detect_gesture_direction(self, gesture_points):
        """
        Detecta la dirección dominante de un gesto usando angle_between_points

        Args:
            gesture_points: Lista de puntos del gesto

        Returns:
            String indicando la dirección: 'horizontal', 'vertical', 'diagonal', 'circular'
        """

        if len(gesture_points) < 3:
            return "unknown"

        angles = []
        for i in range(len(gesture_points) - 1):
            angle = angle_between_points(gesture_points[i], gesture_points[i + 1])
            angles.append(angle)

        # Analizar distribución de ángulos
        angles = np.array(angles)

        # Normalizar ángulos a [0, 2π]
        angles = angles % (2 * np.pi)

        # Determinar patrón dominante
        angle_std = np.std(angles)

        if angle_std < 0.3:  # Ángulos muy consistentes
            avg_angle = np.mean(angles)
            if abs(avg_angle) < 0.3 or abs(avg_angle - np.pi) < 0.3:
                return "horizontal"
            elif (
                abs(avg_angle - np.pi / 2) < 0.3 or abs(avg_angle - 3 * np.pi / 2) < 0.3
            ):
                return "vertical"
            else:
                return "diagonal"
        elif angle_std > 1.5:  # Ángulos muy variables
            return "circular"
        else:
            return "curved"

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
        color_array = enhanced_color_mapping_with_smooth_transitions(normalized)

        return color_array

    def print_info(self):
        """Imprime información inicial del autómata"""
        print("🎯 Ultra Slime Mold Automata Inicializado")
        print(f"📏 Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"🖥️ Ventana: {self.window_width}x{self.window_height}")
        print(f"🎮 FPS objetivo: {FPS}")
        print(f"🔧 Render scale: 1:{self.render_scale}")

    def run(self):
        """Bucle principal del autómata"""
        running = True
        print("🚀 Iniciando simulación...")

        while running:
            # Manejo de eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_window_resize(event.w, event.h)
                else:
                    # Pasar eventos al menú primero
                    if not self.interactive_menu.handle_event(event):
                        self._handle_other_events(event)

            # Actualizar simulación si no está pausada
            if not self.paused and self.frame_count % self.skip_frames == 0:
                self.update_grid_ultra_fast()

            # Renderizar
            self.draw_ultra_fast()

            # Actualizar pantalla
            pygame.display.flip()

            # Control de FPS
            self.clock.tick(FPS)
            self.frame_count += 1

        print("👋 Simulación terminada")

    def update_grid_ultra_fast(self):
        """Actualización ultra rápida usando scipy y numba"""
        # Convolución ultra optimizada
        convolved = ndimage.convolve(
            self.grid,
            self.settings.filters[self.current_filter],
            mode="wrap",
            output=np.float32,
        )

        # Activación vectorizada JIT compilada
        activated = activation_vectorized(convolved)
        self.grid = np.clip(activated, 0.0, 1.0, out=self.grid)
        self.generation += 1

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

        # Dibujar panel de información si está activado
        if self.show_info:
            self.draw_info_panel()

        # Dibujar preview del borrador
        mouse_pos = pygame.mouse.get_pos()
        self.draw_eraser_preview(self.back_buffer, mouse_pos)

        # Dibujar menú interactivo
        self.interactive_menu.draw(self.back_buffer)

        # Copiar buffer trasero a pantalla
        self.screen.blit(self.back_buffer, (0, 0))

    def draw_info_panel(self):
        """Dibuja el panel de información optimizado con cache"""
        current_time = pygame.time.get_ticks()

        # Actualizar FPS cada 500ms
        if current_time - self.last_fps_update > 500:
            self.cached_fps = int(self.clock.get_fps())
            self.last_fps_update = current_time

        # Información actual
        current_info = {
            "fps": self.cached_fps,
            "generation": self.generation,
            "zoom": self.zoom_factor,
            "render_scale": self.render_scale,
            "eraser_size": self.eraser_size,
            "skip_frames": self.skip_frames,
            "paused": self.paused,
        }

        # Verificar si necesitamos actualizar el cache
        if (
            self.info_cache_dirty
            or self.cached_info_surface is None
            or current_info != self.last_info_data
        ):
            # Crear nueva superficie de información
            info_lines = [
                f"FPS: {current_info['fps']}",
                f"Generación: {current_info['generation']}",
                f"Zoom: {current_info['zoom']:.1f}x",
                f"Calidad: 1:{current_info['render_scale']}",
                f"Borrador: {current_info['eraser_size']}px",
                f"Skip: {current_info['skip_frames']} frames",
                f"Estado: {'PAUSADO' if current_info['paused'] else 'EJECUTANDO'}",
            ]

            # Calcular dimensiones del panel
            line_height = 22
            panel_height = len(info_lines) * line_height + 20
            panel_width = 220

            # Crear superficie con transparencia
            self.cached_info_surface = pygame.Surface(
                (panel_width, panel_height), pygame.SRCALPHA
            )

            # Fondo semi-transparente
            background_color = (15, 20, 30, 180)
            self.cached_info_surface.fill(background_color)

            # Dibujar texto
            for i, line in enumerate(info_lines):
                color = (
                    (220, 220, 200) if not current_info["paused"] else (255, 200, 200)
                )
                text_surface = self.small_font.render(line, True, color)
                self.cached_info_surface.blit(text_surface, (10, 10 + i * line_height))

            # Actualizar cache
            self.last_info_data = current_info.copy()
            self.info_cache_dirty = False

        # Dibujar desde cache
        info_x = self.window_width - self.cached_info_surface.get_width() - 10
        info_y = 10
        self.back_buffer.blit(self.cached_info_surface, (info_x, info_y))

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
            # Borrador continuo mientras se mantiene presionado
            if self.mouse_pressed and not self.dragging:
                self.erase_at_mouse_position(event.pos)
            elif self.dragging:
                # Arrastre para mover el grid
                current_mouse_pos = pygame.mouse.get_pos()
                dx = current_mouse_pos[0] - self.drag_start_pos[0]
                dy = current_mouse_pos[1] - self.drag_start_pos[1]

                # Actualizar offset basado en el movimiento del mouse
                self.zoom_offset_x = self.drag_start_offset[0] - dx
                self.zoom_offset_y = self.drag_start_offset[1] - dy

                # Aplicar límites para el offset
                self.clamp_zoom_offset()

        elif event.type == pygame.MOUSEWHEEL:
            # Zoom con rueda del mouse mejorado
            old_zoom = self.zoom_factor

            # Velocidad de zoom adaptiva (más lenta en zoom alto)
            base_speed = 0.1
            zoom_speed = base_speed * (1 + self.zoom_factor * 0.1)

            if event.y > 0:  # Rueda hacia arriba - zoom in
                self.zoom_factor = min(MAX_ZOOM, self.zoom_factor + zoom_speed)
            elif event.y < 0:  # Rueda hacia abajo - zoom out
                self.zoom_factor = max(MIN_ZOOM, self.zoom_factor - zoom_speed)

            # Redondear para evitar valores extraños
            self.zoom_factor = round(self.zoom_factor, 2)

            # Aplicar límites
            self.clamp_zoom_offset()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
                status = "Pausado" if self.paused else "Reanudado"
                print(f"⏸️ {status}")

            elif event.key == pygame.K_m:
                self.interactive_menu.visible = not self.interactive_menu.visible
                status = "Visible" if self.interactive_menu.visible else "Oculto"
                print(f"🎮 Menú: {status}")

            elif event.key == pygame.K_h:
                self.print_info()

            elif event.key == pygame.K_ESCAPE:
                return False

        return True
