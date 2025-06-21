# main.py
# Versión ultra optimizada para grids extremadamente grandes

import sys

import numpy as np
import pygame
from numba import jit
from scipy import ndimage

# Parámetros para ventana y grilla grandes
WIDTH, HEIGHT = 1400, 1000
GRID_SIZE = 1000
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 60  # Reducido para estabilidad con 1M células
RENDER_SCALE = 2  # Renderizar cada N pixels para mayor velocidad


# Función de activación ultra optimizada con Numba JIT
@jit(nopython=True, fastmath=True)
def activation_jit(x):
    """Función de activación JIT compilada"""
    return -1.0 / (0.89 * x * x + 1.0) + 1.0


@jit(nopython=True, fastmath=True)
def activation_vectorized(arr):
    """Activación vectorizada JIT"""
    result = np.empty_like(arr)
    flat_arr = arr.flat
    flat_result = result.flat
    for i in range(arr.size):
        x = flat_arr[i]
        flat_result[i] = -1.0 / (0.89 * x * x + 1.0) + 1.0
    return result


# Filtros optimizados
filters = {
    "default": np.array(
        [[0.8, -0.85, 0.8], [-0.85, -0.2, -0.85], [0.8, -0.85, 0.8]], dtype=np.float32
    ),
    "growth": np.array(
        [[0.68, -0.9, 0.68], [-0.9, -0.66, -0.9], [0.68, -0.9, 0.68]], dtype=np.float32
    ),
    "waves": np.array(
        [[0.565, -0.716, 0.565], [-0.716, 0.627, -0.716], [0.565, -0.716, 0.565]],
        dtype=np.float32,
    ),
    "maze": np.array(
        [[-0.766, -0.854, -0.766], [-0.854, 2.5, -0.854], [-0.766, -0.854, -0.766]],
        dtype=np.float32,
    ),
    "coral": np.array(
        [[0.12, -0.45, 0.12], [-0.45, 0.78, -0.45], [0.12, -0.45, 0.12]],
        dtype=np.float32,
    ),
}


class UltraSlimeMold:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Ultra Slime Mold Automata - Grande")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

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
        self.min_zoom = 0.1
        self.max_zoom = 20.0  # Zoom mucho más allá del grid size

        # Control de arrastre para navegación
        self.dragging = False
        self.drag_start_pos = (0, 0)
        self.drag_start_offset = (0, 0)

        # Optimizaciones extremas
        self.render_scale = RENDER_SCALE  # Escala de renderizado dinámica
        self.skip_frames = 2  # Renderizar cada N frames
        self.frame_count = 0
        self.last_fps_update = 0
        self.cached_fps = 0

        # Pre-compilar funciones JIT
        print("⚡ Pre-compilando funciones JIT...")
        dummy = np.random.rand(10, 10).astype(np.float32)
        activation_vectorized(dummy)
        print("✅ JIT compilado")

        self.print_info()

    def clamp_zoom_offset(self):
        """Limita el offset del zoom para evitar salirse de los límites mejorado"""
        # Para zoom < 1 (zoom out), permitir offset para centrar
        if self.zoom_factor < 1.0:
            # Cuando hay zoom out, permitir centrar la imagen
            max_offset_x = (WIDTH - WIDTH * self.zoom_factor) / 2
            max_offset_y = (HEIGHT - HEIGHT * self.zoom_factor) / 2
            self.zoom_offset_x = max(
                -max_offset_x, min(max_offset_x, self.zoom_offset_x)
            )
            self.zoom_offset_y = max(
                -max_offset_y, min(max_offset_y, self.zoom_offset_y)
            )
        else:
            # Para zoom >= 1, permitir navegar por toda la imagen ampliada
            max_offset_x = WIDTH * (self.zoom_factor - 1)
            max_offset_y = HEIGHT * (self.zoom_factor - 1)
            # Permitir offset negativo para mayor libertad de navegación
            self.zoom_offset_x = max(
                -WIDTH, min(max_offset_x + WIDTH, self.zoom_offset_x)
            )
            self.zoom_offset_y = max(
                -HEIGHT, min(max_offset_y + HEIGHT, self.zoom_offset_y)
            )

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

    def create_color_surface_optimized(self):
        """Crea superficie de colores ultra optimizada con downsampling"""
        # Downsample la grilla para renderizado más rápido
        if self.render_scale > 1:
            # Reducir resolución para renderizado
            downsampled = self.grid[:: self.render_scale, :: self.render_scale]
            render_size = downsampled.shape[0]
        else:
            downsampled = self.grid
            render_size = GRID_SIZE

        # Pre-calcular colores usando vectorización NumPy
        color_array = np.zeros((render_size, render_size, 3), dtype=np.uint8)

        # Mapeo ultra optimizado usando operaciones vectorizadas
        # Convertir a uint8 directamente con operaciones matemáticas
        scaled_values = (downsampled * 255).astype(np.uint8)

        # Color mapping más simple pero efectivo
        # Azul para valores bajos, verde-amarillo para medios, blanco para altos
        color_array[:, :, 0] = scaled_values  # Rojo
        color_array[:, :, 1] = scaled_values  # Verde
        color_array[:, :, 2] = np.minimum(255, scaled_values * 1.2).astype(
            np.uint8
        )  # Azul más intenso

        return color_array

    def draw_ultra_fast(self):
        """Renderizado ultra rápido con downsampling y zoom mejorado"""
        color_array = self.create_color_surface_optimized()

        # Crear superficie desde array
        surf = pygame.surfarray.make_surface(color_array.swapaxes(0, 1))

        # Calcular el nuevo tamaño con zoom
        zoomed_width = int(WIDTH * self.zoom_factor)
        zoomed_height = int(HEIGHT * self.zoom_factor)

        # Escalar la superficie con zoom
        scaled_surf = pygame.transform.scale(surf, (zoomed_width, zoomed_height))

        # Calcular la posición de dibujo considerando el offset
        draw_x = -self.zoom_offset_x
        draw_y = -self.zoom_offset_y

        # Dibujar la superficie escalada
        self.screen.blit(scaled_surf, (draw_x, draw_y))

    def draw_info_panel(self):
        """Panel de información optimizado con cache"""
        if not self.show_info:
            return

        current_time = pygame.time.get_ticks()

        # Actualizar FPS solo cada 500ms
        if current_time - self.last_fps_update > 500:
            self.cached_fps = self.clock.get_fps()
            self.last_fps_update = current_time

        info_lines = [
            f"Filtro: {self.current_filter}",
            f"Estado: {'PAUSADO' if self.paused else 'CORRIENDO'}",
            f"Gen: {self.generation}",
            f"FPS: {self.cached_fps:.1f}",
            f"Grilla: {GRID_SIZE}x{GRID_SIZE}",
            f"Células: {GRID_SIZE * GRID_SIZE:,}",
            f"Escala: 1:{self.render_scale}",
            f"Zoom: {self.zoom_factor:.1f}x",
            f"Offset: ({int(self.zoom_offset_x)}, {int(self.zoom_offset_y)})",
            f"Modo: {'🖱️Borrar' if self.mouse_pressed else '🖱️Arrastrar' if self.dragging else '👀Ver'} | Tamaño: {self.eraser_size}px",
        ]

        # Panel más compacto
        panel_height = len(info_lines) * 20 + 15
        panel_surf = pygame.Surface((280, panel_height))
        panel_surf.set_alpha(200)
        panel_surf.fill((0, 0, 0))

        y_offset = 8
        for line in info_lines:
            text_surf = self.font.render(line, True, (255, 255, 255))
            panel_surf.blit(text_surf, (8, y_offset))
            y_offset += 20

        self.screen.blit(panel_surf, (10, 10))

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

                    # Calcular el punto de zoom (donde está el mouse)
                    zoom_ratio = self.zoom_factor / old_zoom

                    # Ajustar offset para mantener el punto del mouse fijo
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
                elif key in [
                    pygame.K_1,
                    pygame.K_2,
                    pygame.K_3,
                    pygame.K_4,
                    pygame.K_5,
                ]:
                    filter_map = {
                        pygame.K_1: "default",
                        pygame.K_2: "growth",
                        pygame.K_3: "waves",
                        pygame.K_4: "maze",
                        pygame.K_5: "coral",
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
        print("R - 🔄 Reiniciar")
        print("SPACE - ⏸️/▶️ Pausar/Reanudar")
        print("I - 📊 Info on/off")
        print("+ - 📈 Mejor calidad (menos FPS)")
        print("- - 📉 Menor calidad (más FPS)")
        print("F - 🎬 Cambiar frame skip")
        print("E - 🖌️ Cambiar tamaño borrador")
        print("Z - 🔍 Reset zoom")
        print("CLIC IZQ - 🖱️ Borrar (mantener presionado)")
        print("SHIFT+CLIC - 🖱️ Arrastrar vista")
        print("CLIC DER - 🖱️ Arrastrar vista")
        print("RUEDA - 🔍 Zoom in/out (0.1x - 20x)")
        print("1-5 - 🎛️ Cambiar filtros")
        print("ESC - 🚪 Salir")
        print("=" * 45)
        print("💡 TIP: Usa +/- para ajustar rendimiento")

    def run(self):
        """Bucle principal ultra optimizado con frame skipping"""
        running = True

        print("🚀 Iniciando simulación ultra optimizada...")

        while running:
            running = self.handle_input()

            # Actualizar lógica siempre
            if not self.paused:
                self.update_grid_ultra_fast()

            # Renderizar solo cada N frames para mejor rendimiento
            self.frame_count += 1
            if self.frame_count % self.skip_frames == 0:
                self.screen.fill((0, 0, 0))
                self.draw_ultra_fast()
                self.draw_info_panel()
                pygame.display.flip()

            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main():
    """Función principal para ejecutar la simulación"""
    automata = UltraSlimeMold()
    automata.run()


if __name__ == "__main__":
    main()
