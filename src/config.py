"""
Конфигурация эксперимента FNO для эллиптического уравнения.

Все параметры собраны здесь для удобства управления экспериментом.
"""

import numpy as np
from pathlib import Path

class Config:
    # ============================================================
    # ФИЗИЧЕСКАЯ ПОСТАНОВКА ЗАДАЧИ
    # ============================================================
    
    # Область: Ω = [0, 1]²
    DOMAIN_X_MIN = 0.0
    DOMAIN_X_MAX = 1.0
    DOMAIN_Y_MIN = 0.0
    DOMAIN_Y_MAX = 1.0
    
    # Правая часть: f(x) = 1
    RHS_VALUE = 1.0
    
    # ============================================================
    # ПАРАМЕТРИЗАЦИЯ КОЭФФИЦИЕНТА a(x)
    # ============================================================
    
    # Базовое значение коэффициента диффузии
    A_BASE = 2.0
    
    # Амплитуда флуктуаций для GRF
    A_SIGMA = 0.8
    
    # Число мод в разложении GRF
    GRF_N_MODES = 12
    
    # Минимальное значение a(x) для обеспечения эллиптичности
    # После генерации проверяем: если a < A_MIN, клиппируем
    A_MIN = 0.5
    
    # ============================================================
    # ДИСКРЕТИЗАЦИЯ FEM (DOLFINx)
    # ============================================================
    
    # Разрешение mesh для FEM
    # На GTX 1050 Ti можем позволить достаточно мелкую сетку
    FEM_NX = 128  # число ячеек по x
    FEM_NY = 128  # число ячеек по y
    
    # Степень полиномов конечных элементов
    FEM_DEGREE = 1  # P1 элементы (линейные)
    
    # ============================================================
    # РАЗРЕШЕНИЕ ДЛЯ FNO
    # ============================================================
    
    # FNO работает на uniform grid. 
    # Можем взять меньше, чем FEM, для экономии памяти.
    FNO_RESOLUTION = 64  # 64×64 grid
    
    # ============================================================
    # DATASET
    # ============================================================
    
    # Размеры датасета
    # Для GTX 1050 Ti 4GB выбираем консервативно
    N_TRAIN = 800
    N_VAL = 100
    N_TEST = 100
    # N_TRAIN = 50
    # N_VAL = 10
    # N_TEST = 10
    N_TOTAL = N_TRAIN + N_VAL + N_TEST  # 1000
    
    # Путь для сохранения данных
    DATA_DIR = Path("data")
    DATA_FILE = DATA_DIR / "elliptic_dataset.npz"
    
    # ============================================================
    # АРХИТЕКТУРА FNO
    # ============================================================
    
    # Ширина скрытого представления
    FNO_WIDTH = 32  # Достаточно для 4GB, можно увеличить до 48-64 при желании
    
    # Число Fourier layers
    FNO_N_LAYERS = 4
    
    # Число Fourier modes (k_max)
    # Учитываются только низкие частоты
    FNO_MODES = 12  # первые 12 мод по каждой оси
    
    # Добавлять ли координаты как input channels
    FNO_USE_COORDS = True  # если True, вход (a, x, y), иначе только a
    
    # ============================================================
    # ОБУЧЕНИЕ
    # ============================================================
    
    # Batch size — критично для VRAM
    # На GTX 1050 Ti 4GB с resolution=64, width=32 можем взять ~16-20
    BATCH_SIZE = 16
    
    # Число эпох
    N_EPOCHS = 200
    
    # Learning rate
    LEARNING_RATE = 1e-3
    
    # Learning rate scheduler (опционально)
    USE_LR_SCHEDULER = True
    LR_SCHEDULE_MILESTONES = [100, 150]  # уменьшаем LR на этих эпохах
    LR_SCHEDULE_GAMMA = 0.5  # коэффициент уменьшения
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 30  # если val loss не улучшается 30 эпох
    
    # ============================================================
    # CHECKPOINT И ЛОГИ
    # ============================================================
    
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    RESULTS_DIR = Path("results")
    
    # Частота сохранения checkpoint
    CHECKPOINT_EVERY = 20  # каждые 20 эпох
    
    # ============================================================
    # МЕТРИКИ
    # ============================================================
    
    # Относительная L2 ошибка:
    # ||u_pred - u_true||_L2 / ||u_true||_L2
    METRIC_NAMES = ["rel_l2", "mse", "max_error"]
    
    # ============================================================
    # ВИЗУАЛИЗАЦИЯ
    # ============================================================
    
    # Число примеров для визуализации
    N_VIS_SAMPLES = 5
    
    # DPI для сохранения изображений
    FIG_DPI = 150
    
    # ============================================================
    # SEED ДЛЯ ВОСПРОИЗВОДИМОСТИ
    # ============================================================
    
    SEED = 42
    
    # ============================================================
    # РЕЖИМЫ РАБОТЫ (для debugging)
    # ============================================================
    
    # Можно создать маленькую конфигурацию для быстрой проверки
    DEBUG_MODE = False
    
    @classmethod
    def apply_debug_mode(cls):
        """Уменьшить все размеры для быстрой отладки."""
        cls.DEBUG_MODE = True
        cls.N_TRAIN = 20
        cls.N_VAL = 5
        cls.N_TEST = 5
        cls.N_TOTAL = 30
        cls.N_EPOCHS = 10
        cls.FNO_WIDTH = 16
        cls.FNO_MODES = 8
        cls.BATCH_SIZE = 4
        print("[DEBUG MODE] Конфигурация уменьшена для быстрой отладки")
    
    @classmethod
    def setup_directories(cls):
        """Создать необходимые директории."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Вывести текущую конфигурацию."""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА")
        print("=" * 60)
        print(f"Dataset: train={cls.N_TRAIN}, val={cls.N_VAL}, test={cls.N_TEST}")
        print(f"FEM resolution: {cls.FEM_NX}×{cls.FEM_NY}")
        print(f"FNO resolution: {cls.FNO_RESOLUTION}×{cls.FNO_RESOLUTION}")
        print(f"FNO architecture: width={cls.FNO_WIDTH}, modes={cls.FNO_MODES}, layers={cls.FNO_N_LAYERS}")
        print(f"Training: epochs={cls.N_EPOCHS}, batch_size={cls.BATCH_SIZE}, lr={cls.LEARNING_RATE}")
        print(f"Coefficient: base={cls.A_BASE}, sigma={cls.A_SIGMA}, modes={cls.GRF_N_MODES}")
        print("=" * 60)