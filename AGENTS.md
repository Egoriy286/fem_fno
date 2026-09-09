# AGENTS.md

## Состояние проекта

**КРИТИЧНО**: Реализация **НЕ ЗАВЕРШЕНА**. Все файлы ниже были **запланированы**, но **не созданы**.

Существует только:
- `pyproject.toml` - минимальная структура пакета
- `src/fem_fno/__init__.py` - placeholder
- Пустые директории: `src/fem/`, `src/fno/`

**НЕТ в репозитории**:
- `config.py`, `generate_data.py`, `dataset.py`, `fno.py`, `train.py`, `evaluate.py`, `visualize.py`
- Любая реализация FEM или FNO
- Генерированные данные, чекпоинты, результаты

## Архитектура (целевая)

### Математическая задача
Параметрическое эллиптическое PDE:
```
-∇·(a(x)∇u(x)) = f(x) в Ω=[0,1]²
u = 0 на ∂Ω
```

- **Параметр**: `a(x)` через Gaussian Random Field
- **Ground truth**: DOLFINx FEM (128×128 mesh, P1 элементы)
- **FNO**: 64×64 uniform grid

### Целевая структура файлов

```
src/
├── config.py           # Все параметры (физика, FEM, FNO, обучение)
├── generate_data.py    # DOLFINx: GRF → FEM решение → интерполяция
├── dataset.py          # PyTorch Dataset, нормализация
├── fno.py             # JAX: SpectralConv2d → FourierLayer → FNO2d
├── train.py           # Trainer, metrics, checkpointing
├── evaluate.py        # Тестирование, детальный анализ
└── visualize.py       # Графики результатов
```

### Зависимости (через uv)

**Критичные**:
- `dolfinx` - FEM решение (требует conda или специальная сборка)
- `jax[cuda]` - FNO обучение (GPU-specific)
- `flax`, `optax` - нейросетевая архитектура
- `numpy`, `matplotlib`, `tqdm`

**Окружение**: Python 3.13+, GTX 1050 Ti 4GB (оптимизация памяти обязательна)

## Команды выполнения (целевые)

```powershell
# Установка (после реализации)
uv sync

# Pipeline (порядок важен)
uv run python -m src.generate_data  # 1. Генерация обучающих данных
uv run python -m src.train          # 2. Обучение FNO
uv run python -m src.evaluate       # 3. Тестирование
uv run python -m src.visualize      # 4. Визуализация результатов
```

## Критичные детали реализации

### DOLFINx (generate_data.py)
- **Mesh**: `dolfinx.mesh.create_unit_square(comm, 128, 128)`
- **Элементы**: P1 (линейные)
- **Weak form**: `∫ a(x)∇u·∇v dx = ∫ f v dx`
- **BC**: `dolfinx.fem.dirichletbc` на всей границе
- **Интерполяция**: FEM → uniform 64×64 grid через numpy interpolation

### FNO архитектура (fno.py)
```
Lifting (1→32) → [Fourier Layer]×4 → Projection (32→1)

Fourier Layer:
  SpectralConv2d: FFT → truncate k_max=12 → W_k → IFFT
  + skip connection (conv 1×1)
  + GELU activation
```

### Гиперпараметры (config.py)
- **Dataset**: 800 train / 100 val / 100 test
- **FNO**: width=32, modes=12, layers=4
- **Обучение**: batch_size=16, Adam, lr=1e-3 → 1e-4 schedule
- **Метрики**: relative L2 error, MSE, max error

## Частые ошибки (избегать)

1. **DOLFINx установка**: НЕ через `pip install dolfinx` - нужна conda или специальная сборка
2. **JAX GPU**: Требует `jax[cuda]` с соответствующей версией CUDA
3. **Память GPU**: 4GB лимит → обязателен batch_size=16, gradient accumulation если нужно
4. **Координаты**: FNO требует (x, y) координаты как дополнительные каналы
5. **Нормализация**: Обязательна для a(x) и u(x) перед обучением
6. **Метрики**: Relative L2 error = ||u_pred - u_true||_L2 / ||u_true||_L2

## Референс

- FNO paper: `2010.08895v3.pdf` в корне репозитория
- Математическая постановка: weak formulation для эллиптических PDE
- Код на русском: все комментарии, docstrings, README

## Следующие шаги для реализации

1. Создать `config.py` с полной конфигурацией
2. Реализовать `generate_data.py` - проверить DOLFINx установку ПЕРВЫМ
3. Реализовать `dataset.py`, `fno.py`, `train.py` последовательно
4. Протестировать на малом датасете (10 образцов) перед полной генерацией
5. Мониторить память GPU во время обучения

## Полезные проверки

```powershell
# DOLFINx доступен?
uv run python -c "import dolfinx; print(dolfinx.__version__)"

# JAX видит GPU?
uv run python -c "import jax; print(jax.devices())"

# Структура данных корректна?
# После generate_data должно быть:
# data/train.npz, data/val.npz, data/test.npz
# Каждый содержит: 'a_coeffs', 'u_solutions'
```
