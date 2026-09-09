"""
Реализация Fourier Neural Operator (FNO) на JAX.

Архитектура:
    Input a(x) → Lifting → [Fourier Layers]×L → Projection → Output u(x)

Ключевые компоненты:
    1. Spectral Convolution: операция в Fourier space
    2. Fourier Layer: spectral conv + skip connection + activation
    3. FNO: полная архитектура

Математика:
    Spectral convolution:
        (Kv)(x) = F⁻¹(R · F(v))(x)
    
    где F — Fourier transform, R — обучаемые веса в spectral domain.
    
    Fourier layer:
        v_{l+1} = σ(W·v_l + K(v_l))
    
    где W — skip connection, σ — нелинейность (GELU).
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Sequence
import numpy as np

from config import Config


class SpectralConv2d(nn.Module):
    """
    Spectral Convolution в 2D.
    
    Операция:
        1. FFT: v(x,y) → v̂(k_x, k_y)
        2. Truncate к первым k_max модам
        3. Умножение на обучаемые веса: R · v̂
        4. Padding обратно к полному размеру
        5. IFFT: v̂ → v(x,y)
    
    Атрибуты:
        in_channels: число входных каналов
        out_channels: число выходных каналов
        modes1, modes2: число Fourier мод по каждой оси (k_max)
    """
    
    in_channels: int
    out_channels: int
    modes1: int  # число мод по x
    modes2: int  # число мод по y
    
    @nn.compact
    def __call__(self, x):
        """
        Параметры:
            x: (batch, height, width, in_channels)
        
        Возвращает:
            out: (batch, height, width, out_channels)
        """
        batch_size, height, width, _ = x.shape
        
        # ========================================
        # 1. Forward FFT
        # ========================================
        # Переводим в частотное представление
        # JAX FFT работает с комплексными числами
        # rfft2 для real input → half-complex output (экономия памяти)
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # (batch, height, width//2+1, in_channels)
        
        # ========================================
        # 2. Обучаемые веса в Fourier space
        # ========================================
        # Инициализируем комплексные веса для низких мод
        # Используем Xavier/Glorot инициализацию
        scale = 1.0 / (self.in_channels * self.out_channels)
        
        # Веса для нижних левых мод [0:modes1, 0:modes2]
        weights1_real = self.param(
            'weights1_real',
            nn.initializers.normal(stddev=scale),
            (self.modes1, self.modes2, self.in_channels, self.out_channels)
        )
        weights1_imag = self.param(
            'weights1_imag',
            nn.initializers.normal(stddev=scale),
            (self.modes1, self.modes2, self.in_channels, self.out_channels)
        )
        
        # Веса для верхних левых мод [-modes1:, 0:modes2]
        # Нужны для корректной обработки высоких частот
        weights2_real = self.param(
            'weights2_real',
            nn.initializers.normal(stddev=scale),
            (self.modes1, self.modes2, self.in_channels, self.out_channels)
        )
        weights2_imag = self.param(
            'weights2_imag',
            nn.initializers.normal(stddev=scale),
            (self.modes1, self.modes2, self.in_channels, self.out_channels)
        )
        
        # Формируем комплексные веса
        weights1 = weights1_real + 1j * weights1_imag
        weights2 = weights2_real + 1j * weights2_imag
        
        # ========================================
        # 3. Умножение в Fourier space
        # ========================================
        # Инициализируем выходной массив в Fourier space
        out_ft = jnp.zeros(
            (batch_size, height, width // 2 + 1, self.out_channels),
            dtype=jnp.complex64
        )
        
        # Multiply lower modes [0:modes1, 0:modes2]
        # Это операция матричного умножения для каждой моды
        # Размерности:
        #   x_ft[:, :modes1, :modes2, :]: (batch, modes1, modes2, in_channels)
        #   weights1: (modes1, modes2, in_channels, out_channels)
        # Нужно сделать einsum для batch matrix multiplication
        
        out_ft_lower = jnp.einsum(
            'bxyi,xyio->bxyo',
            x_ft[:, :self.modes1, :self.modes2, :],
            weights1
        )
        out_ft = out_ft.at[:, :self.modes1, :self.modes2, :].set(out_ft_lower)
        
        # Multiply upper modes [-modes1:, 0:modes2]
        out_ft_upper = jnp.einsum(
            'bxyi,xyio->bxyo',
            x_ft[:, -self.modes1:, :self.modes2, :],
            weights2
        )
        out_ft = out_ft.at[:, -self.modes1:, :self.modes2, :].set(out_ft_upper)
        
        # ========================================
        # 4. Inverse FFT
        # ========================================
        # Возвращаемся в пространственное представление
        x_out = jnp.fft.irfft2(out_ft, s=(height, width), axes=(1, 2))
        
        return x_out


class FourierLayer(nn.Module):
    """
    Fourier Layer = Spectral Conv + Skip Connection + Activation.
    
    Операция:
        v_{l+1} = σ(W·v_l + K(v_l))
    
    где:
        K — spectral convolution
        W — обычная 1×1 convolution (skip connection)
        σ — GELU activation
    
    Атрибуты:
        width: число каналов (in = out для residual)
        modes1, modes2: число Fourier мод
    """
    
    width: int
    modes1: int
    modes2: int
    
    @nn.compact
    def __call__(self, x):
        """
        Параметры:
            x: (batch, height, width, width)
        
        Возвращает:
            out: (batch, height, width, width)
        """
        # ========================================
        # Path 1: Spectral Convolution
        # ========================================
        # Глобальная операция через Fourier transform
        x1 = SpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes1=self.modes1,
            modes2=self.modes2
        )(x)
        
        # ========================================
        # Path 2: Skip Connection
        # ========================================
        # Локальная операция: point-wise (1x1) convolution
        # Это эквивалентно Dense layer, примененному к каждой точке
        x2 = nn.Dense(features=self.width)(x)
        
        # ========================================
        # Combine and Activate
        # ========================================
        # Суммируем оба пути
        x_combined = x1 + x2
        
        # Применяем нелинейность
        # GELU: x·Φ(x), где Φ — CDF стандартного нормального распределения
        # GELU более гладкая чем ReLU, часто работает лучше для FNO
        x_out = nn.gelu(x_combined)
        
        return x_out


class FNO2d(nn.Module):
    """
    Полная архитектура Fourier Neural Operator для 2D задач.
    
    Структура:
        Input → Lifting → [Fourier Layer]×n_layers → Projection → Output
    
    Атрибуты:
        modes1, modes2: число Fourier мод
        width: ширина скрытого представления
        n_layers: число Fourier layers
        in_channels: число входных каналов (1 для a(x) или 3 для a+coords)
        out_channels: число выходных каналов (1 для u(x))
    """
    
    modes1: int
    modes2: int
    width: int
    n_layers: int
    in_channels: int = 3  # a(x), x, y
    out_channels: int = 1  # u(x)
    
    @nn.compact
    def __call__(self, x):
        """
        Параметры:
            x: (batch, height, width, in_channels)
        
        Возвращает:
            out: (batch, height, width, out_channels)
        """
        # ========================================
        # 1. Lifting (P)
        # ========================================
        # Проецируем входные каналы в высокомерное пространство
        # a(x) ∈ R^{in_channels} → v_0 ∈ R^{width}
        # Это позволяет FNO работать в богатом пространстве признаков
        x = nn.Dense(features=self.width)(x)
        
        # ========================================
        # 2. Fourier Layers
        # ========================================
        # Последовательно применяем L Fourier layers
        # Каждый layer обучает трансформацию в spectral domain
        for _ in range(self.n_layers):
            x = FourierLayer(
                width=self.width,
                modes1=self.modes1,
                modes2=self.modes2
            )(x)
        
        # ========================================
        # 3. Projection (Q)
        # ========================================
        # Проецируем обратно из скрытого пространства в выходное
        # v_L ∈ R^{width} → u ∈ R^{out_channels}
        # Обычно несколько Dense layers для более гладкой проекции
        x = nn.Dense(features=self.width)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.out_channels)(x)
        
        return x
    
    def count_params(self):
        """
        Подсчёт числа параметров модели.
        
        Полезно для оценки memory footprint.
        """
        # Это метод будет вызван после инициализации
        # В JAX параметры хранятся отдельно от модуля
        pass


def create_fno_model(config=None):
    """
    Factory function для создания FNO модели.
    
    Параметры:
        config: Config объект (опционально)
    
    Возвращает:
        model: FNO2d instance
    """
    if config is None:
        config = Config
    
    # Определяем число входных каналов
    in_channels = 3 if config.FNO_USE_COORDS else 1
    
    model = FNO2d(
        modes1=config.FNO_MODES,
        modes2=config.FNO_MODES,
        width=config.FNO_WIDTH,
        n_layers=config.FNO_N_LAYERS,
        in_channels=in_channels,
        out_channels=1
    )
    
    return model


def initialize_model(model, rng, input_shape):
    """
    Инициализация параметров модели.
    
    В JAX/Flax параметры инициализируются отдельно через dummy forward pass.
    
    Параметры:
        model: FNO2d instance
        rng: JAX random key
        input_shape: (batch, height, width, channels)
    
    Возвращает:
        params: словарь параметров модели
    """
    # Dummy input для инициализации
    dummy_input = jnp.ones(input_shape)
    
    # Инициализация через init
    params = model.init(rng, dummy_input)
    
    return params


def count_parameters(params):
    """
    Подсчёт общего числа параметров.
    
    Параметры:
        params: словарь параметров из model.init()
    
    Возвращает:
        n_params: общее число параметров
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


if __name__ == "__main__":
    # Тестирование архитектуры
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ FNO АРХИТЕКТУРЫ")
    print("=" * 60)
    
    # Создание модели
    model = create_fno_model()
    print(f"\nМодель создана:")
    print(f"  Modes: {Config.FNO_MODES}")
    print(f"  Width: {Config.FNO_WIDTH}")
    print(f"  Layers: {Config.FNO_N_LAYERS}")
    
    # Инициализация
    rng = random.PRNGKey(0)
    batch_size = 2
    input_shape = (batch_size, Config.FNO_RESOLUTION, Config.FNO_RESOLUTION, 3)
    
    print(f"\nИнициализация с input shape: {input_shape}")
    params = initialize_model(model, rng, input_shape)
    
    # Подсчёт параметров
    n_params = count_parameters(params)
    print(f"\nОбщее число параметров: {n_params:,}")
    print(f"Примерный размер модели: {n_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Тестовый forward pass
    print("\nТестовый forward pass...")
    dummy_input = jnp.ones(input_shape)
    output = model.apply(params, dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Проверка размерностей
    assert output.shape == (batch_size, Config.FNO_RESOLUTION, Config.FNO_RESOLUTION, 1)
    print("\n✓ Архитектура работает корректно")
    
    # Оценка памяти для обучения
    print("\n" + "=" * 60)
    print("ОЦЕНКА ТРЕБОВАНИЙ К ПАМЯТИ")
    print("=" * 60)
    
    # Model parameters
    model_memory = n_params * 4 / 1024**2  # MB
    
    # Activations (rough estimate)
    # Для каждого Fourier layer храним промежуточные активации
    activation_per_sample = (
        Config.FNO_RESOLUTION ** 2 * Config.FNO_WIDTH * 
        Config.FNO_N_LAYERS * 4  # bytes per float32
    ) / 1024**2
    
    batch_memory = activation_per_sample * Config.BATCH_SIZE
    
    # Optimizer state (Adam: 2 buffers per parameter)
    optimizer_memory = model_memory * 2
    
    total_memory = model_memory + batch_memory + optimizer_memory
    
    print(f"Model parameters: {model_memory:.1f} MB")
    print(f"Batch activations: {batch_memory:.1f} MB")
    print(f"Optimizer state: {optimizer_memory:.1f} MB")
    print(f"Total (estimated): {total_memory:.1f} MB")
    print(f"\nGTX 1050 Ti VRAM: 4096 MB")
    print(f"Available for data: ~{4096 - total_memory:.1f} MB")
    
    if total_memory < 3000:
        print("\n✓ Конфигурация безопасна для GTX 1050 Ti 4GB")
    else:
        print("\n⚠ Может быть недостаточно памяти. Уменьшите batch_size или width.")