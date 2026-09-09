"""
Dataset loader для обучения FNO.

Загружает данные, сгенерированные DOLFINx, и подготавливает их для JAX.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from config import Config


class EllipticDataset:
    """
    Dataset для эллиптического PDE.
    
    Структура:
        - Input: a(x) — коэффициент, shape (H, W)
        - Target: u(x) — решение, shape (H, W)
    
    Опционально добавляем координатную сетку как дополнительные каналы.
    """
    
    def __init__(self, split="train", normalize=True, use_coords=True):
        """
        Параметры:
            split: "train", "val" или "test"
            normalize: нормализовать ли данные
            use_coords: добавлять ли координаты (x, y) как каналы
        """
        self.split = split
        self.normalize = normalize
        self.use_coords = use_coords
        
        # Загрузка данных
        self._load_data()
        
        # Нормализация
        if self.normalize:
            self._compute_normalization_stats()
            self._normalize_data()
        
        # Добавление координат
        if self.use_coords:
            self._add_coordinates()
        
        print(f"[{split.upper()}] Загружено {len(self)} образцов")
        print(f"  Input shape:  {self.inputs.shape}")
        print(f"  Target shape: {self.targets.shape}")
    
    def _load_data(self):
        """Загрузка из .npz файла."""
        if not Config.DATA_FILE.exists():
            raise FileNotFoundError(
                f"Датасет не найден: {Config.DATA_FILE}\n"
                "Запустите generate_data.py сначала."
            )
        
        data = np.load(Config.DATA_FILE)
        
        # Выбор split
        if self.split == "train":
            self.a = data["a_train"]
            self.u = data["u_train"]
        elif self.split == "val":
            self.a = data["a_val"]
            self.u = data["u_val"]
        elif self.split == "test":
            self.a = data["a_test"]
            self.u = data["u_test"]
        else:
            raise ValueError(f"Неизвестный split: {self.split}")
        
        # Добавляем channel dimension: (N, H, W) → (N, H, W, 1)
        self.inputs = self.a[..., np.newaxis]
        self.targets = self.u[..., np.newaxis]
    
    def _compute_normalization_stats(self):
        """
        Вычисление статистик для нормализации.
        
        Важно: используем только train split для вычисления статистик,
        чтобы избежать data leakage.
        """
        if self.split == "train":
            # Вычисляем на train
            self.a_mean = self.a.mean()
            self.a_std = self.a.std()
            self.u_mean = self.u.mean()
            self.u_std = self.u.std()
            
            # Сохраняем для val/test
            self._save_normalization_stats()
        else:
            # Загружаем из train
            self._load_normalization_stats()
    
    def _save_normalization_stats(self):
        """Сохранение статистик нормализации."""
        stats_file = Config.DATA_DIR / "normalization_stats.npz"
        np.savez(
            stats_file,
            a_mean=self.a_mean,
            a_std=self.a_std,
            u_mean=self.u_mean,
            u_std=self.u_std
        )
    
    def _load_normalization_stats(self):
        """Загрузка статистик нормализации."""
        stats_file = Config.DATA_DIR / "normalization_stats.npz"
        if not stats_file.exists():
            raise FileNotFoundError(
                "Нормализационные статистики не найдены.\n"
                "Создайте train dataset сначала."
            )
        
        stats = np.load(stats_file)
        self.a_mean = float(stats["a_mean"])
        self.a_std = float(stats["a_std"])
        self.u_mean = float(stats["u_mean"])
        self.u_std = float(stats["u_std"])
    
    def _normalize_data(self):
        """
        Стандартная нормализация (z-score).
        
        Нормализация помогает обучению:
            - стабилизирует градиенты
            - ускоряет сходимость
        """
        # Нормализация input
        self.inputs = (self.inputs - self.a_mean) / (self.a_std + 1e-8)
        
        # Нормализация target
        self.targets = (self.targets - self.u_mean) / (self.u_std + 1e-8)
    
    def denormalize_output(self, u_normalized):
        """
        Обратная нормализация для получения физических значений.
        
        Используется после inference FNO.
        """
        if not self.normalize:
            return u_normalized
        
        return u_normalized * self.u_std + self.u_mean
    
    def _add_coordinates(self):
        """
        Добавление нормализованных координат (x, y) как дополнительных каналов.
        
        Input становится (N, H, W, 3) вместо (N, H, W, 1):
            channel 0: a(x, y)
            channel 1: x ∈ [0, 1]
            channel 2: y ∈ [0, 1]
        
        Координаты помогают FNO "знать", где он находится в пространстве.
        Особенно полезно для non-periodic BC.
        """
        N, H, W, C = self.inputs.shape
        
        # Создаём координатную сетку
        x = np.linspace(0, 1, H)
        y = np.linspace(0, 1, W)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # Расширяем для батча
        xx = np.tile(xx[np.newaxis, ..., np.newaxis], (N, 1, 1, 1))
        yy = np.tile(yy[np.newaxis, ..., np.newaxis], (N, 1, 1, 1))
        
        # Конкатенация
        self.inputs = np.concatenate([self.inputs, xx, yy], axis=-1)
    
    def __len__(self):
        """Число образцов в dataset."""
        return len(self.targets)
    
    def __getitem__(self, idx):
        """
        Получение одного образца.
        
        Возвращает:
            input: (H, W, C) где C=3 если use_coords, иначе C=1
            target: (H, W, 1)
        """
        return self.inputs[idx], self.targets[idx]
    
    def get_batch(self, indices):
        """
        Получение батча по индексам.
        
        Параметры:
            indices: массив индексов
        
        Возвращает:
            inputs: (B, H, W, C)
            targets: (B, H, W, 1)
        """
        return self.inputs[indices], self.targets[indices]
    
    def get_all_data(self):
        """Получение всех данных сразу."""
        return self.inputs, self.targets
    
    def to_jax(self):
        """Преобразование в JAX arrays."""
        self.inputs = jnp.array(self.inputs)
        self.targets = jnp.array(self.targets)
        return self


def create_batches(dataset_size, batch_size, shuffle=True, rng=None):
    """
    Генератор индексов батчей.
    
    Параметры:
        dataset_size: размер dataset
        batch_size: размер батча
        shuffle: перемешивать ли данные
        rng: numpy random generator
    
    Yields:
        indices: массив индексов для батча
    """
    indices = np.arange(dataset_size)
    
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    
    # Генерация батчей
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        yield indices[start_idx:end_idx]


if __name__ == "__main__":
    # Тест загрузки
    Config.setup_directories()
    
    print("Тестирование dataset loader...\n")
    
    train_dataset = EllipticDataset("train", normalize=True, use_coords=True)
    val_dataset = EllipticDataset("val", normalize=True, use_coords=True)
    test_dataset = EllipticDataset("test", normalize=True, use_coords=True)
    
    print("\nПример образца:")
    inp, tgt = train_dataset[0]
    print(f"Input shape:  {inp.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Input channels: {inp.shape[-1]}")
    
    print("\nПроверка батчей:")
    rng = np.random.default_rng(42)
    for i, batch_indices in enumerate(create_batches(len(train_dataset), Config.BATCH_SIZE, shuffle=True, rng=rng)):
        inputs, targets = train_dataset.get_batch(batch_indices)
        print(f"Batch {i}: inputs {inputs.shape}, targets {targets.shape}")
        if i >= 2:
            break
    
    print("\n✓ Dataset loader работает корректно")