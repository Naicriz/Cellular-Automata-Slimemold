"""Sistema de cache inteligente para optimizar rendimiento"""

import time
from typing import Any, Dict, Optional, Tuple


class CacheManager:
    """Gestor de cache inteligente con TTL y límites de tamaño"""

    def __init__(self, max_size: int = 100, default_ttl: float = 60.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float, float]] = (
            {}
        )  # key: (value, timestamp, ttl)
        self.access_times: Dict[str, float] = {}  # Para LRU

    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache"""
        if key not in self.cache:
            return None

        value, timestamp, ttl = self.cache[key]
        current_time = time.time()

        # Verificar si ha expirado
        if current_time - timestamp > ttl:
            self.remove(key)
            return None

        # Actualizar tiempo de acceso para LRU
        self.access_times[key] = current_time
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Almacena un valor en el cache"""
        if ttl is None:
            ttl = self.default_ttl

        current_time = time.time()

        # Si el cache está lleno, remover el elemento menos usado recientemente
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        self.cache[key] = (value, current_time, ttl)
        self.access_times[key] = current_time

    def remove(self, key: str) -> bool:
        """Remueve un elemento del cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            return True
        return False

    def clear(self) -> None:
        """Limpia todo el cache"""
        self.cache.clear()
        self.access_times.clear()

    def cleanup_expired(self) -> int:
        """Limpia elementos expirados y retorna cantidad removida"""
        current_time = time.time()
        expired_keys = []

        for key, (_, timestamp, ttl) in self.cache.items():
            if current_time - timestamp > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self.remove(key)

        return len(expired_keys)

    def _evict_lru(self) -> None:
        """Remueve el elemento menos usado recientemente"""
        if not self.access_times:
            return

        # Encontrar la clave con el tiempo de acceso más antiguo
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.remove(lru_key)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        current_time = time.time()
        expired_count = 0

        for _, timestamp, ttl in self.cache.values():
            if current_time - timestamp > ttl:
                expired_count += 1

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expired_items": expired_count,
            "utilization": len(self.cache) / self.max_size * 100,
        }


class SurfaceCache:
    """Cache específico para superficies pygame con compresión de claves"""

    def __init__(self, max_surfaces: int = 20):
        self.cache_manager = CacheManager(max_size=max_surfaces, default_ttl=30.0)

    def get_surface(
        self, grid_hash: int, render_scale: int, zoom_factor: float
    ) -> Optional[Any]:
        """Obtiene una superficie cacheada"""
        key = f"{grid_hash}_{render_scale}_{zoom_factor:.1f}"
        return self.cache_manager.get(key)

    def cache_surface(
        self, grid_hash: int, render_scale: int, zoom_factor: float, surface: Any
    ) -> None:
        """Cachea una superficie"""
        key = f"{grid_hash}_{render_scale}_{zoom_factor:.1f}"
        self.cache_manager.set(key, surface, ttl=30.0)

    def clear(self) -> None:
        """Limpia el cache de superficies"""
        self.cache_manager.clear()

    def cleanup(self) -> int:
        """Limpia superficies expiradas"""
        return self.cache_manager.cleanup_expired()


class UICache:
    """Cache específico para elementos de UI"""

    def __init__(self):
        self.cache_manager = CacheManager(max_size=50, default_ttl=120.0)

    def get_ui_element(self, element_id: str, state_hash: str) -> Optional[Any]:
        """Obtiene un elemento UI cacheado"""
        key = f"ui_{element_id}_{state_hash}"
        return self.cache_manager.get(key)

    def cache_ui_element(self, element_id: str, state_hash: str, element: Any) -> None:
        """Cachea un elemento UI"""
        key = f"ui_{element_id}_{state_hash}"
        self.cache_manager.set(key, element, ttl=120.0)

    def invalidate_element(self, element_id: str) -> int:
        """Invalida todos los elementos con un ID específico"""
        removed = 0
        keys_to_remove = [
            key
            for key in self.cache_manager.cache.keys()
            if key.startswith(f"ui_{element_id}_")
        ]

        for key in keys_to_remove:
            if self.cache_manager.remove(key):
                removed += 1

        return removed
