# main.py
# Versión ultra optimizada para grids extremadamente grandes

import sys

import numpy as np
import pygame
from numba import jit
from scipy import ndimage

# Parámetros para ventana y grilla
WIDTH, HEIGHT = 1400, 1000
GRID_SIZE = 1000
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 60
RENDER_SCALE = 1  # Renderizar cada N pixels para mayor velocidad

# Optimizaciones de rendimiento
NUMBA_CACHE = True  # Cache JIT compilations
FAST_MATH = True  # Optimizaciones matemáticas agresivas


# Función de activación ultra optimizada con Numba JIT
@jit(nopython=True, fastmath=True, cache=True)
def activation_jit(x):
    """Función de activación JIT compilada"""
    return -1.0 / (0.89 * x * x + 1.0) + 1.0


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def activation_vectorized(arr):
    """Activación vectorizada JIT con paralelización"""
    result = np.empty_like(arr)
    flat_arr = arr.flat
    flat_result = result.flat
    for i in range(arr.size):
        x = flat_arr[i]
        flat_result[i] = -1.0 / (0.89 * x * x + 1.0) + 1.0
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
                color_array[y, x, 1] = np.uint8(val * 25 + 5)   # Menos verde
                color_array[y, x, 2] = np.uint8(val * 10)       # Muy poco azul
            elif val <= 0.35:
                # Marrón anaranjado oscuro
                color_array[y, x, 0] = np.uint8(val * 100 + 40)  # Predominio rojo
                color_array[y, x, 1] = np.uint8(val * 80 + 20)   # Algo de verde
                color_array[y, x, 2] = np.uint8(val * 20)        # Poco azul
            elif val <= 0.55:
                # Naranja medio brillante
                color_array[y, x, 0] = np.uint8(val * 120 + 100) # Rojo fuerte
                color_array[y, x, 1] = np.uint8(val * 100 + 60)  # Verde moderado
                color_array[y, x, 2] = np.uint8(val * 30 + 10)   # Azul mínimo
            elif val <= 0.75:
                # Amarillo anaranjado
                color_array[y, x, 0] = 255                        # Rojo máximo
                g_val = val * 140 + 100
                color_array[y, x, 1] = np.uint8(255 if g_val > 255 else g_val)
                color_array[y, x, 2] = np.uint8(val * 40 + 20)   # Algo de azul para calidez
            elif val <= 0.9:
                # Amarillo brillante cálido
                color_array[y, x, 0] = 255                        # Rojo máximo
                color_array[y, x, 1] = 255                        # Verde máximo
                color_array[y, x, 2] = np.uint8(val * 80 + 50)   # Más azul para amarillo cálido
            else:
                # Amarillo-blanco incandescente con tinte cálido
                color_array[y, x, 0] = 255                        # Rojo máximo
                color_array[y, x, 1] = 255                        # Verde máximo
                b_val = val * 140 + 100  # Más azul para blanco cálido
                color_array[y, x, 2] = np.uint8(255 if b_val > 255 else b_val)

    return color_array


# Filtros para convolución del grid
filters = {
    "default": np.array(
        [[0.8, -0.85, 0.8], [-0.85, -0.2, -0.85], [0.8, -0.85, 0.8]], dtype=np.float32
    ),

}


class UltraSlimeMold:
    """Clase principal para el autómata celular Slime Mold"""
    def __init__(self):
        pygame.init()

        # Configurar pantalla con flags anti-flicker
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        pygame.display.set_caption("Ultra Slime Mold Automata - Grande")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

        # Buffer de superficie para renderizado suave
        self.back_buffer = pygame.Surface((WIDTH, HEIGHT))
        self.last_surface = None
        self.surface_ready = False

        # Estado
        self.grid = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)
        self.current_filter = "default"
        self.paused = False
        self.show_info = True
        self.generation = 0

        # Control del mouse para borrar
        self.mouse_pressed = False
        self.eraser_size = 5  # Radio del borrador

        # Control de zoom y navegación
        self.zoom_factor = 1.0
        self.zoom_offset_x = 0
        self.zoom_offset_y = 0
        self.min_zoom = (
            1.0  # No permitir zoom out más allá del tamaño completo del grid
        )
        self.max_zoom = 20.0  # Zoom mucho más allá del grid size

        # Control de arrastre para navegación
        self.dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_offset = (0, 0)

        # Control de efectos visuales
        self.glow_effects = True  # Efectos de brillo
        self.particle_effects = True  # Efectos de partículas (pueden ser costosos)

        # Optimizaciones extremas
        self.render_scale = RENDER_SCALE  # Escala de renderizado dinámica
        self.skip_frames = 1  # Cuántos frames saltar para mayor rendimiento
        self.frame_count = 0
        self.last_fps_update = 0
        self.cached_fps = 0

        # Cache de superficies optimizado para máximo rendimiento
        self.surface_cache = {}
        self.last_grid_hash = None
        self.cached_base_surface = None
        self.zoom_cache = {}
        self.bright_cells_cache = {}

        # Cache específico para el panel de información (anti-parpadeo)
        self.info_panel_cache = None
        self.last_info_update = 0
        self.cached_info_data = None

        # Anti-flicker settings
        self.vsync_enabled = True
        self.frame_interpolation = True
        self.smooth_transitions = True

        # Pre-compilar funciones JIT
        print("⚡ Pre-compilando funciones JIT...")
        dummy = np.random.rand(10, 10).astype(np.float32)
        activation_vectorized(dummy)
        create_color_mapping_jit(dummy)
        print("✅ JIT compilado y optimizado")

        self.print_info()

    def clamp_zoom_offset(self):
        """Limita el offset del zoom para mantener el grid siempre visible"""
        # Para zoom >= 1, permitir navegar por toda la imagen ampliada
        # pero manteniendo siempre algo del grid visible
        max_offset_x = WIDTH * (self.zoom_factor - 1)
        max_offset_y = HEIGHT * (self.zoom_factor - 1)

        # Límites estrictos: no permitir salirse del área del grid
        self.zoom_offset_x = max(0, min(max_offset_x, self.zoom_offset_x))
        self.zoom_offset_y = max(0, min(max_offset_y, self.zoom_offset_y))

    def erase_at_mouse_position(self, mouse_pos):
        """Borra células en la posición del mouse"""
        mouse_x, mouse_y = mouse_pos

        # Convertir coordenadas de pantalla a coordenadas de grilla considerando zoom
        # Ajustar por el zoom y offset
        adjusted_x = (mouse_x + self.zoom_offset_x) / self.zoom_factor
        adjusted_y = (mouse_y + self.zoom_offset_y) / self.zoom_factor

        grid_x = int((adjusted_x / WIDTH) * GRID_SIZE)
        grid_y = int((adjusted_y / HEIGHT) * GRID_SIZE)

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
        self.back_buffer.fill((40, 40, 40))

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
                surf, (WIDTH, HEIGHT)
            )
            self.last_grid_hash = current_grid_checksum
        elif hasattr(self, "cached_base_surface"):
            # Reutilizar superficie cacheada
            pass
        else:
            # Fallback si no hay cache
            color_array = self.create_color_surface_bioluminescent()
            surf = pygame.surfarray.make_surface(color_array.swapaxes(0, 1))
            self.cached_base_surface = pygame.transform.smoothscale(
                surf, (WIDTH, HEIGHT)
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

                zoomed_width = int(WIDTH * self.zoom_factor)
                zoomed_height = int(HEIGHT * self.zoom_factor)
                scaled_surf = pygame.transform.smoothscale(
                    self.cached_base_surface, (zoomed_width, zoomed_height)
                )
                self.zoom_cache = {"key": zoom_key, "surface": scaled_surf}
            else:
                scaled_surf = self.zoom_cache["surface"]
        else:
            scaled_surf = self.cached_base_surface

        # Calcular la posición de dibujo considerando el offset
        draw_x = -self.zoom_offset_x
        draw_y = -self.zoom_offset_y

        # Dibujar la superficie escalada al buffer trasero
        self.back_buffer.blit(scaled_surf, (draw_x, draw_y))

        # Agregar efectos solo si el rendimiento lo permite
        if self.glow_effects and self.cached_fps > 30:
            self.add_glow_effect_to_buffer(scaled_surf, draw_x, draw_y)

        # Marcar superficie como lista
        self.surface_ready = True
        self.last_surface = scaled_surf

    def draw_info_panel(self):
        """Panel de información ultra optimizado con cache anti-parpadeo"""
        if not self.show_info:
            return

        current_time = pygame.time.get_ticks()

        # Actualizar FPS solo cada 1000ms para mayor estabilidad
        if current_time - self.last_fps_update > 1000:
            self.cached_fps = self.clock.get_fps()
            self.last_fps_update = current_time

        # Crear datos del panel de información
        current_info_data = {
            "filter": self.current_filter,
            "paused": self.paused,
            "generation": self.generation,
            "fps": self.cached_fps,
            "render_scale": self.render_scale,
            "zoom": self.zoom_factor,
            "offset_x": int(self.zoom_offset_x),
            "offset_y": int(self.zoom_offset_y),
            "mouse_pressed": self.mouse_pressed,
            "dragging": self.dragging,
            "eraser_size": self.eraser_size,
            "glow_effects": self.glow_effects,
            "particle_effects": self.particle_effects,
        }

        # Solo regenerar el panel si los datos han cambiado significativamente
        info_changed = (
            self.cached_info_data is None
            or self.cached_info_data != current_info_data
            or current_time - self.last_info_update
            > 2000  # Forzar actualización cada 2 segundos
        )

        if info_changed:
            info_lines = [
                f"Filtro: {current_info_data['filter']}",
                f"Estado: {'PAUSADO' if current_info_data['paused'] else 'CORRIENDO'}",
                f"Gen: {current_info_data['generation']}",
                f"FPS: {current_info_data['fps']:.1f}",
                f"Grilla: {GRID_SIZE}x{GRID_SIZE}",
                f"Células: {GRID_SIZE * GRID_SIZE:,}",
                f"Escala: 1:{current_info_data['render_scale']}",
                f"Zoom: {current_info_data['zoom']:.1f}x",
                f"Offset: ({current_info_data['offset_x']}, {current_info_data['offset_y']})",
                f"Modo: {'🖱️Borrar' if current_info_data['mouse_pressed'] else '🖱️Arrastrar' if current_info_data['dragging'] else '👀Ver'} | Tamaño: {current_info_data['eraser_size']}px",
                f"✨Brillo: {'ON' if current_info_data['glow_effects'] else 'OFF'} | 🎆Partículas: {'ON' if current_info_data['particle_effects'] else 'OFF'}",
                f"⚡Anti-flicker: ON | 📺V-Sync: {'ON' if self.vsync_enabled else 'OFF'}",
            ]

            # Crear nueva superficie del panel
            panel_height = len(info_lines) * 20 + 15
            panel_surf = pygame.Surface((320, panel_height), pygame.SRCALPHA)
            panel_surf.set_alpha(220)  # Ligeramente más opaco para mejor legibilidad
            panel_surf.fill((0, 0, 0))

            # Dibujar texto con borde sutil para mejor legibilidad
            y_offset = 8
            for line in info_lines:
                # Sombra de texto para mejor contraste
                shadow_surf = self.font.render(line, True, (0, 0, 0))
                panel_surf.blit(shadow_surf, (9, y_offset + 1))

                # Texto principal
                text_surf = self.font.render(line, True, (255, 255, 255))
                panel_surf.blit(text_surf, (8, y_offset))
                y_offset += 20

            # Cachear el panel generado
            self.info_panel_cache = panel_surf
            self.cached_info_data = current_info_data
            self.last_info_update = current_time

        # Dibujar el panel cacheado al buffer trasero
        if self.info_panel_cache:
            self.back_buffer.blit(self.info_panel_cache, (10, 10))

    def handle_input(self):
        """Manejo de entrada optimizado"""
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
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
                if key == pygame.K_r:
                    self.grid = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)
                    self.generation = 0
                    print("🔄 Grilla reiniciada")
                elif key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"⏸️ {'Pausado' if self.paused else '▶️ Reanudado'}")
                elif key == pygame.K_i:
                    self.show_info = not self.show_info
                elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
                    # Aumentar calidad de renderizado
                    if self.render_scale > 1:
                        self.render_scale = max(1, self.render_scale - 1)
                        print(f"📈 Escala de render: 1:{self.render_scale}")
                elif key == pygame.K_MINUS:
                    # Disminuir calidad de renderizado para más FPS
                    self.render_scale = min(8, self.render_scale + 1)
                    print(f"📉 Escala de render: 1:{self.render_scale}")
                elif key == pygame.K_f:
                    # Cambiar frame skipping
                    self.skip_frames = 3 - self.skip_frames + 1  # Alterna entre 1, 2, 3
                    print(f"🎬 Frame skip: {self.skip_frames}")
                elif key == pygame.K_e:
                    # Cambiar tamaño del borrador
                    self.eraser_size = (self.eraser_size % 10) + 1  # Cicla entre 1-10
                    print(f"🖌️ Tamaño borrador: {self.eraser_size}")
                elif key == pygame.K_z:
                    # Reset zoom
                    self.zoom_factor = 1.0
                    self.zoom_offset_x = 0
                    self.zoom_offset_y = 0
                    print("🔍 Zoom reseteado")
                elif key == pygame.K_g:
                    # Toggle efectos de brillo
                    self.glow_effects = not self.glow_effects
                    print(
                        f"✨ Efectos de brillo: {'ON' if self.glow_effects else 'OFF'}"
                    )
                elif key == pygame.K_p:
                    # Toggle efectos de partículas
                    self.particle_effects = not self.particle_effects
                    print(
                        f"🎆 Efectos de partículas: {'ON' if self.particle_effects else 'OFF'}"
                    )
                elif key == pygame.K_v:
                    # Toggle V-Sync
                    self.vsync_enabled = not self.vsync_enabled
                    print(f"📺 V-Sync: {'ON' if self.vsync_enabled else 'OFF'}")
                elif key in [
                    pygame.K_1
                ]:
                    filter_map = {
                        pygame.K_1: "default"
                    }
                    self.current_filter = filter_map[key]
                    print(f"🎛️ Filtro: {self.current_filter}")
                elif key == pygame.K_ESCAPE:
                    return False
        return True

    def print_info(self):
        """Información inicial"""
        print("\n🧬 === ULTRA SLIME MOLD AUTOMATA ===")
        print(f"📏 Ventana: {WIDTH}x{HEIGHT} pixels")
        print(f"🔲 Grilla: {GRID_SIZE}x{GRID_SIZE} células")
        print(f"🎯 Células totales: {GRID_SIZE * GRID_SIZE:,}")
        print(
            f"⚡ Render escala: 1:{self.render_scale if hasattr(self, 'render_scale') else RENDER_SCALE}"
        )
        print("\n🎮 CONTROLES:")
        print("ALGUNOS(VARIOS) CONTROLES PUEDEN NO ESTAR FUNCIONANDO CORRECTAMENTE")
        print("R - 🔄 Reiniciar")
        print("SPACE - ⏸️/▶️ Pausar/Reanudar")
        print("I - 📊 Info on/off")
        print("+ - 📈 Mejor calidad (menos FPS)")
        print("- - 📉 Menor calidad (más FPS)")
        print("F - 🎬 Cambiar frame skip")
        print("E - 🖌️ Cambiar tamaño borrador")
        print("Z - 🔍 Reset zoom")
        print("G - ✨ Toggle efectos de brillo")
        print("P - 🎆 Toggle efectos de partículas")
        print("V - 📺 Toggle V-Sync (anti-flicker)")
        print("CLIC IZQ - 🖱️ Borrar (mantener presionado)")
        print("SHIFT+CLIC - 🖱️ Arrastrar vista")
        print("CLIC DER - 🖱️ Arrastrar vista")
        print("RUEDA - 🔍 Zoom in/out (1.0x - 20x)")  # Actualizado el rango mínimo
        print("1-5 - 🎛️ Cambiar filtros")
        print("ESC - 🚪 Salir")
        print("=" * 50)
        print("💡 TIP: Usa G para efectos bioluminiscentes")
        print("🔍 TIP: El zoom está limitado al área del grid")
        print("📺 TIP: V-Sync reduce el flicker pero puede limitar FPS")

    def add_glow_effect_to_buffer(self, surf, draw_x, draw_y):
        """Efectos de brillo bioluminiscente ultra suave con gradientes naturales"""
        # Solo si zoom es suficiente y cada N frames (aún menos frecuente para ultra suavidad)
        if self.zoom_factor < 0.5 or self.generation % 8 != 0:
            return

        # Cache de células brillantes - actualizar aún menos frecuentemente
        if (
            not hasattr(self, "bright_cells_cache")
            or "y" not in self.bright_cells_cache
            or self.generation % 25 == 0  # Actualizar cada 25 frames para ultra suavidad
        ):

            if self.render_scale > 1:
                downsampled = self.grid[:: self.render_scale, :: self.render_scale]
            else:
                downsampled = self.grid[::2, ::2]

            # Umbral aún más bajo para más efectos graduales
            bright_cells = np.where(downsampled > 0.65)
            self.bright_cells_cache = {
                "y": bright_cells[0],
                "x": bright_cells[1],
                "downsampled_shape": downsampled.shape,
            }

        bright_y = self.bright_cells_cache["y"]
        bright_x = self.bright_cells_cache["x"]
        downsampled_shape = self.bright_cells_cache["downsampled_shape"]

        if len(bright_y) > 0:
            # Más efectos para gradiente ultra suave
            max_effects = min(120, len(bright_y))
            step = max(1, len(bright_y) // max_effects)

            # Pre-calcular valores comunes con animación ultra lenta
            surf_width = surf.get_width()
            surf_height = surf.get_height()
            # Animación 5x más lenta que antes
            generation_factor = self.generation * 0.015

            for i in range(0, len(bright_y), step):
                y, x = bright_y[i], bright_x[i]

                # Convertir coordenadas
                screen_x = int((x / downsampled_shape[1]) * surf_width + draw_x)
                screen_y = int((y / downsampled_shape[0]) * surf_height + draw_y)

                # Verificar límites
                if not (0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT):
                    continue

                # Obtener intensidad real de la célula para gradiente natural
                real_intensity = self.grid[
                    min(int(y * self.render_scale), GRID_SIZE - 1),
                    min(int(x * self.render_scale), GRID_SIZE - 1)
                ]

                # Pulso ultra suave y lento basado en posición
                pulse_base = np.sin(generation_factor + x * 0.02 + y * 0.02)
                pulse = (pulse_base + 1) / 2  # Normalizar a 0-1
                # Hacer el pulso más sutil
                pulse = 0.8 + (pulse * 0.2)  # Entre 0.8 y 1.0 para menos variación

                # Radio variable basado en intensidad real
                base_radius = 3 + (real_intensity * 3)
                glow_radius = int(base_radius + pulse * 1.5)

                # Crear efecto de gradiente múltiple según intensidad
                self._draw_gradient_glow(
                    screen_x, screen_y, glow_radius, real_intensity, pulse
                )

    def _draw_gradient_glow(self, screen_x, screen_y, radius, intensity, pulse):
        """Dibuja un efecto de brillo con gradiente suave - Paleta amarillo-anaranjada"""
        if radius < 1:
            return

        # Colores base según intensidad con gradiente amarillo-anaranjado cálido
        if intensity > 0.9:
            # Amarillo-blanco incandescente con tinte cálido
            core_color = (255, 255, 240)
            mid1_color = (255, 255, 200)
            mid2_color = (255, 240, 160)
            mid3_color = (255, 220, 120)
            outer_color = (255, 200, 100)
        elif intensity > 0.8:
            # Amarillo brillante cálido
            core_color = (255, 255, 200)
            mid1_color = (255, 240, 160)
            mid2_color = (255, 220, 120)
            mid3_color = (255, 200, 80)
            outer_color = (255, 180, 60)
        elif intensity > 0.7:
            # Amarillo-naranja
            core_color = (255, 220, 120)
            mid1_color = (255, 200, 100)
            mid2_color = (255, 180, 80)
            mid3_color = (255, 160, 60)
            outer_color = (255, 140, 40)
        else:
            # Naranja suave
            core_color = (255, 180, 80)
            mid1_color = (255, 160, 60)
            mid2_color = (255, 140, 40)
            mid3_color = (240, 120, 30)
            outer_color = (220, 100, 20)

        # Modificar intensidad con pulso ultra suave
        pulse_intensity = 0.8 + (pulse * 0.2)  # Entre 0.8 y 1.0

        # Dibujar múltiples capas para gradiente ultra suave (más capas)
        layers = [
            (radius, outer_color, int(20 * pulse_intensity)),        # Halo externo
            (radius * 0.85, mid3_color, int(25 * pulse_intensity)),  # Capa 4
            (radius * 0.7, mid2_color, int(35 * pulse_intensity)),   # Capa 3
            (radius * 0.55, mid1_color, int(45 * pulse_intensity)),  # Capa 2
            (radius * 0.35, core_color, int(60 * pulse_intensity))   # Núcleo
        ]

        for layer_radius, color, alpha in layers:
            if layer_radius >= 1:
                layer_radius = int(layer_radius)
                glow_surf = pygame.Surface(
                    (layer_radius * 2, layer_radius * 2), pygame.SRCALPHA
                )
                
                # Color con alpha para transparencia
                layer_color = (*color, alpha)
                
                pygame.draw.circle(
                    glow_surf, layer_color, 
                    (layer_radius, layer_radius), layer_radius
                )
                
                self.back_buffer.blit(
                    glow_surf,
                    (screen_x - layer_radius, screen_y - layer_radius),
                    special_flags=pygame.BLEND_ADD,
                )

    def add_particle_effects_to_buffer(self):
        """Efectos de partículas bioluminiscentes ultra suaves al buffer trasero"""
        # Solo si hay suficiente actividad y zoom adecuado
        # Ultra menos frecuente para reducir flicker y hacer más sutil
        if self.generation % 25 == 0 and self.zoom_factor > 0.8:
            highly_active = np.where(self.grid > 0.95)

            if len(highly_active[0]) > 0:
                # Seleccionar algunas células al azar para efectos de partículas
                num_particles = min(
                    15, len(highly_active[0])
                )  # Menos partículas para reducir flicker
                indices = np.random.choice(
                    len(highly_active[0]), num_particles, replace=False
                )

                for idx in indices:
                    y, x = highly_active[0][idx], highly_active[1][idx]

                    # Convertir a coordenadas de pantalla
                    screen_x = int(
                        (x / GRID_SIZE) * WIDTH * self.zoom_factor - self.zoom_offset_x
                    )
                    screen_y = int(
                        (y / GRID_SIZE) * HEIGHT * self.zoom_factor - self.zoom_offset_y
                    )

                    # Verificar que esté visible
                    if -50 <= screen_x <= WIDTH + 50 and -50 <= screen_y <= HEIGHT + 50:
                        # Crear pequeña "explosión" de partículas amarillo-anaranjadas suavizadas
                        for _ in range(2):  # Menos partículas por célula
                            offset_x = np.random.randint(-3, 4)  # Rango más pequeño
                            offset_y = np.random.randint(-3, 4)
                            particle_color = (
                                255,
                                200,
                                80,
                                120,
                            )  # Color amarillo-anaranjado cálido

                            particle_surf = pygame.Surface(
                                (2, 2), pygame.SRCALPHA
                            )  # Más pequeñas
                            pygame.draw.circle(particle_surf, particle_color, (1, 1), 1)
                            self.back_buffer.blit(
                                particle_surf,
                                (screen_x + offset_x, screen_y + offset_y),
                                special_flags=pygame.BLEND_ADD,
                            )

    def run(self):
        """Bucle principal optimizado"""
        running = True
        print("🚀 Iniciando simulación...")

        while running:
            # Manejar eventos de manera más eficiente
            running = self.handle_input()
            if not running:
                break

            # Actualizar lógica de simulación solo si no está pausado
            if not self.paused:
                self.update_grid_ultra_fast()

            # Renderizado optimizado con frame skipping inteligente
            self.frame_count += 1
            should_render = (
                self.frame_count % self.skip_frames == 0
                or self.frame_count % 10 == 0  # Garantizar rendering mínimo
            )

            if should_render:
                # Renderizar al buffer trasero
                self.draw_ultra_fast()

                # Panel de información estable
                self.draw_info_panel()

                # Efectos solo si el rendimiento es bueno
                if (
                    self.particle_effects
                    and self.cached_fps > 30
                    and self.frame_count % 3 == 0
                ):
                    self.add_particle_effects_to_buffer()

                # Transferir buffer a pantalla
                self.screen.blit(self.back_buffer, (0, 0))
                pygame.display.flip()

            # Control de FPS optimizado
            if self.vsync_enabled and self.cached_fps > 45:
                self.clock.tick(FPS)
            else:
                self.clock.tick_busy_loop(FPS)  # Más eficiente para FPS altos

        pygame.quit()
        sys.exit()


def main():
    """Función principal para ejecutar la simulación"""
    automata = UltraSlimeMold()
    automata.run()


if __name__ == "__main__":
    main()
