"""
Обучение FNO модели.

Pipeline:
    1. Загрузка данных
    2. Создание модели
    3. Определение loss function
    4. Training loop с validation
    5. Checkpoint лучшей модели
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax
import numpy as np
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import json

from config import Config
from dataset import EllipticDataset, create_batches
from fno import create_fno_model, initialize_model, count_parameters


class Metrics:
    """Класс для вычисления и хранения метрик."""
    
    @staticmethod
    def relative_l2_error(pred, target):
        """
        Относительная L2 ошибка:
            ||u_pred - u_true||_L2 / ||u_true||_L2
        
        Это основная метрика для оценки FNO.
        """
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        # L2 norm
        diff_norm = jnp.sqrt(jnp.sum((pred_flat - target_flat) ** 2, axis=1))
        target_norm = jnp.sqrt(jnp.sum(target_flat ** 2, axis=1))
        
        # Относительная ошибка
        rel_error = diff_norm / (target_norm + 1e-8)
        
        return jnp.mean(rel_error)
    
    @staticmethod
    def mse(pred, target):
        """Mean Squared Error."""
        return jnp.mean((pred - target) ** 2)
    
    @staticmethod
    def max_error(pred, target):
        """Maximum absolute error."""
        return jnp.max(jnp.abs(pred - target))


def create_loss_fn(model):
    """
    Создание loss function для обучения.
    
    Используем относительную L2 ошибку как основную loss.
    """
    def loss_fn(params, batch_input, batch_target):
        """
        Параметры:
            params: параметры модели
            batch_input: (batch, H, W, C)
            batch_target: (batch, H, W, 1)
        
        Возвращает:
            loss: скаляр
        """
        # Forward pass
        pred = model.apply(params, batch_input)
        
        # Относительная L2 ошибка
        loss = Metrics.relative_l2_error(pred, batch_target)
        
        return loss
    
    return loss_fn


def create_train_step(model, optimizer):
    """
    Создание jit-компилированной функции для одного шага обучения.
    
    Параметры:
        model: FNO модель
        optimizer: optax optimizer
    
    Возвращает:
        train_step: функция для одного шага
    """
    loss_fn = create_loss_fn(model)
    
    @jit
    def train_step(params, opt_state, batch_input, batch_target):
        """
        Один шаг градиентного спуска.
        
        Возвращает:
            params: обновлённые параметры
            opt_state: обновлённое состояние оптимизатора
            loss: значение loss
        """
        # Вычисляем loss и градиенты
        loss, grads = value_and_grad(loss_fn)(params, batch_input, batch_target)
        
        # Обновляем параметры через optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    return train_step


def create_eval_step(model):
    """
    Создание jit-компилированной функции для evaluation.
    
    Возвращает:
        eval_step: функция для вычисления метрик
    """
    @jit
    def eval_step(params, batch_input, batch_target):
        """
        Вычисление метрик без градиентов.
        
        Возвращает:
            metrics: словарь с метриками
        """
        # Forward pass
        pred = model.apply(params, batch_input)
        
        # Вычисляем все метрики
        metrics = {
            'rel_l2': Metrics.relative_l2_error(pred, batch_target),
            'mse': Metrics.mse(pred, batch_target),
            'max_error': Metrics.max_error(pred, batch_target)
        }
        
        return metrics
    
    return eval_step


class Trainer:
    """
    Класс для управления процессом обучения.
    """
    
    def __init__(self, config=None):
        """Инициализация trainer."""
        self.config = config if config is not None else Config
        self.config.setup_directories()
        
        # Random number generator
        self.rng = np.random.default_rng(self.config.SEED)
        self.jax_rng = random.PRNGKey(self.config.SEED)
        
        # Загрузка данных
        print("Загрузка данных...")
        self.train_dataset = EllipticDataset(
            "train", 
            normalize=True, 
            use_coords=self.config.FNO_USE_COORDS
        )
        self.val_dataset = EllipticDataset(
            "val", 
            normalize=True, 
            use_coords=self.config.FNO_USE_COORDS
        )
        
        # Преобразуем в JAX arrays для ускорения
        self.train_dataset.to_jax()
        self.val_dataset.to_jax()
        
        # Создание модели
        print("\nСоздание модели...")
        self.model = create_fno_model(self.config)
        
        # Инициализация параметров
        input_shape = (
            self.config.BATCH_SIZE,
            self.config.FNO_RESOLUTION,
            self.config.FNO_RESOLUTION,
            3 if self.config.FNO_USE_COORDS else 1
        )
        self.params = initialize_model(self.model, self.jax_rng, input_shape)
        
        n_params = count_parameters(self.params)
        print(f"Число параметров: {n_params:,}")
        
        # Создание optimizer
        self.optimizer = self._create_optimizer()
        self.opt_state = self.optimizer.init(self.params)
        
        # Создание train/eval functions
        self.train_step = create_train_step(self.model, self.optimizer)
        self.eval_step = create_eval_step(self.model)
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def _create_optimizer(self):
        """Создание optimizer с learning rate schedule."""
        if self.config.USE_LR_SCHEDULER:
            # Piecewise constant schedule
            # Уменьшаем LR на определённых эпохах
            boundaries = [
                m * (len(self.train_dataset) // self.config.BATCH_SIZE)
                for m in self.config.LR_SCHEDULE_MILESTONES
            ]
            
            values = [self.config.LEARNING_RATE]
            for _ in self.config.LR_SCHEDULE_MILESTONES:
                values.append(values[-1] * self.config.LR_SCHEDULE_GAMMA)
            
            schedule = optax.piecewise_constant_schedule(
                init_value=self.config.LEARNING_RATE,
                boundaries_and_scales={
                    b: self.config.LR_SCHEDULE_GAMMA 
                    for b in boundaries
                }
            )
        else:
            schedule = self.config.LEARNING_RATE
        
        # Adam optimizer — стандартный выбор для FNO
        optimizer = optax.adam(learning_rate=schedule)
        
        return optimizer
    
    def train_epoch(self, epoch):
        """
        Обучение на одной эпохе.
        
        Возвращает:
            avg_loss: средний loss за эпоху
        """
        epoch_losses = []
        
        # Создаём батчи с перемешиванием
        batch_generator = create_batches(
            len(self.train_dataset),
            self.config.BATCH_SIZE,
            shuffle=True,
            rng=self.rng
        )
        
        # Прогресс бар
        n_batches = len(self.train_dataset) // self.config.BATCH_SIZE
        pbar = tqdm(batch_generator, total=n_batches, desc=f"Epoch {epoch+1}")
        
        for batch_indices in pbar:
            # Получаем батч
            batch_input, batch_target = self.train_dataset.get_batch(batch_indices)
            
            # Один шаг обучения
            self.params, self.opt_state, loss = self.train_step(
                self.params,
                self.opt_state,
                batch_input,
                batch_target
            )
            
            # Сохраняем loss
            epoch_losses.append(float(loss))
            
            # Обновляем прогресс бар
            pbar.set_postfix({'loss': f"{float(loss):.6f}"})
        
        avg_loss = np.mean(epoch_losses)
        return avg_loss
    
    def validate(self):
        """
        Validation на val dataset.
        
        Возвращает:
            metrics: словарь с метриками
        """
        all_metrics = {name: [] for name in ['rel_l2', 'mse', 'max_error']}
        
        # Validation без перемешивания
        batch_generator = create_batches(
            len(self.val_dataset),
            self.config.BATCH_SIZE,
            shuffle=False
        )
        
        for batch_indices in batch_generator:
            batch_input, batch_target = self.val_dataset.get_batch(batch_indices)
            
            # Вычисляем метрики
            batch_metrics = self.eval_step(
                self.params,
                batch_input,
                batch_target
            )
            
            # Собираем метрики
            for name in all_metrics:
                all_metrics[name].append(float(batch_metrics[name]))
        
        # Усредняем метрики
        avg_metrics = {
            name: np.mean(values) 
            for name, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Сохранение checkpoint.
        
        Параметры:
            epoch: номер эпохи
            is_best: сохранять ли как лучшую модель
        """
        checkpoint = {
            'epoch': epoch,
            'params': self.params,
            'opt_state': self.opt_state,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': {
                'FNO_MODES': self.config.FNO_MODES,
                'FNO_WIDTH': self.config.FNO_WIDTH,
                'FNO_N_LAYERS': self.config.FNO_N_LAYERS,
                'FNO_RESOLUTION': self.config.FNO_RESOLUTION,
                'FNO_USE_COORDS': self.config.FNO_USE_COORDS,
            }
        }
        
        # Сохраняем текущий checkpoint
        checkpoint_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Сохраняем лучшую модель отдельно
        if is_best:
            best_path = self.config.CHECKPOINT_DIR / "best_model.pkl"
            with open(best_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  → Сохранена лучшая модель (val_loss: {self.best_val_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Загрузка checkpoint для продолжения обучения."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint['params']
        self.opt_state = checkpoint['opt_state']
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint загружен. Продолжение с эпохи {start_epoch}")
        
        return start_epoch
    
    def train(self, resume_from=None):
        """
        Полный цикл обучения.
        
        Параметры:
            resume_from: путь к checkpoint для продолжения обучения
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
        
        print("\n" + "=" * 60)
        print("НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 60)
        self.config.print_config()
        
        total_start_time = time.time()
        
        for epoch in range(start_epoch, self.config.N_EPOCHS):
            epoch_start_time = time.time()
            
            # Обучение
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            val_loss = val_metrics['rel_l2']
            
            # Сохраняем историю
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Время эпохи
            epoch_time = time.time() - epoch_start_time
            
            # Вывод статистики
            print(f"\nEpoch {epoch+1}/{self.config.N_EPOCHS}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MSE:    {val_metrics['mse']:.6f}")
            print(f"  Val Max Err: {val_metrics['max_error']:.6f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Проверка улучшения
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Сохранение checkpoint
            if (epoch + 1) % self.config.CHECKPOINT_EVERY == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping: val loss не улучшается {self.config.EARLY_STOPPING_PATIENCE} эпох")
                break
        
        total_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print(f"Общее время: {total_time/60:.1f} мин")
        print(f"Лучшая эпоха: {self.best_epoch+1}")
        print(f"Лучший val loss: {self.best_val_loss:.6f}")
        
        # Сохраняем историю
        self._save_history()
    
    def _save_history(self):
        """Сохранение истории обучения в JSON."""
        history_path = self.config.LOG_DIR / "training_history.json"
        
        # Преобразуем numpy types в python types для JSON
        history_json = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'val_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in self.history['val_metrics']
            ],
            'best_epoch': int(self.best_epoch),
            'best_val_loss': float(self.best_val_loss)
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"\nИстория сохранена в {history_path}")


if __name__ == "__main__":
    import sys
    
    # Проверка режима отладки
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        Config.apply_debug_mode()
    
    # Создание и запуск trainer
    trainer = Trainer(Config)
    
    # Проверка, есть ли checkpoint для продолжения
    resume_path = None
    if len(sys.argv) > 2 and sys.argv[2] == "--resume":
        resume_path = sys.argv[3] if len(sys.argv) > 3 else Config.CHECKPOINT_DIR / "best_model.pkl"
    
    trainer.train(resume_from=resume_path)