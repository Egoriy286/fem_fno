"""
Evaluation и тестирование обученной FNO модели.

Функции:
    1. Загрузка лучшей модели
    2. Evaluation на test set
    3. Детальный анализ ошибок
    4. Сравнение с FEM baseline
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import json
from tqdm import tqdm

from config import Config
from dataset import EllipticDataset, create_batches
from fno import create_fno_model
from train import Metrics


class Evaluator:
    """
    Класс для evaluation модели на test set.
    """
    
    def __init__(self, checkpoint_path=None, config=None):
        """
        Параметры:
            checkpoint_path: путь к checkpoint (по умолчанию best_model.pkl)
            config: Config объект
        """
        self.config = config if config is not None else Config
        
        # Путь к checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.config.CHECKPOINT_DIR / "best_model.pkl"
        
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint не найден: {self.checkpoint_path}\n"
                "Обучите модель сначала (train.py)"
            )
        
        # Загрузка checkpoint
        print(f"Загрузка модели из {self.checkpoint_path}...")
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint['params']
        self.best_epoch = checkpoint['best_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Модель загружена (эпоха {self.best_epoch+1}, val_loss: {self.best_val_loss:.6f})")
        
        # Создание модели
        self.model = create_fno_model(self.config)
        
        # JIT-компилируем только вычисление модели.
        self._predict = jax.jit(self.model.apply)
        
        # Загрузка test dataset
        print("\nЗагрузка test dataset...")
        self.test_dataset = EllipticDataset(
            "test",
            normalize=True,
            use_coords=self.config.FNO_USE_COORDS
        )
        self.test_dataset.to_jax()
        
        
        print(f"Test dataset: {len(self.test_dataset)} образцов")
    
    # @jax.jit
    def predict(self, params, input_batch):
        """
        Prediction для батча.
        
        Параметры:
            params: параметры модели
            input_batch: (batch, H, W, C)
        
        Возвращает:
            predictions: (batch, H, W, 1)
        """
        return self.model.apply(params, input_batch)
    
    def evaluate_all(self):
        """
        Evaluation на всём test set.
        
        Возвращает:
            results: словарь с результатами
        """
        print("\n" + "=" * 60)
        print("EVALUATION НА TEST SET")
        print("=" * 60)
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        # Батчи для inference
        batch_generator = create_batches(
            len(self.test_dataset),
            self.config.BATCH_SIZE,
            shuffle=False
        )
        
        print("\nИнференс...")
        for batch_indices in tqdm(batch_generator):
            batch_input, batch_target = self.test_dataset.get_batch(batch_indices)
            
            # Prediction
            batch_pred = self.predict(self.params, batch_input)
            
            # Сохраняем
            all_predictions.append(np.array(batch_pred))
            all_targets.append(np.array(batch_target))
            all_inputs.append(np.array(batch_input))
        
        # Объединяем все батчи
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        print(f"Predictions shape: {predictions.shape}")
        
        # Вычисляем метрики
        print("\nВычисление метрик...")
        
        # Denormalize для физических значений
        predictions_denorm = self.test_dataset.denormalize_output(predictions)
        targets_denorm = self.test_dataset.denormalize_output(targets)
        
        # Метрики на нормализованных данных
        rel_l2_norm = float(Metrics.relative_l2_error(
            jnp.array(predictions),
            jnp.array(targets)
        ))
        
        # Метрики на денормализованных данных (физические значения)
        rel_l2_phys = float(Metrics.relative_l2_error(
            jnp.array(predictions_denorm),
            jnp.array(targets_denorm)
        ))
        
        mse = float(Metrics.mse(
            jnp.array(predictions_denorm),
            jnp.array(targets_denorm)
        ))
        
        max_err = float(Metrics.max_error(
            jnp.array(predictions_denorm),
            jnp.array(targets_denorm)
        ))
        
        # Вычисляем относительные ошибки для каждого образца
        sample_errors = []
        for i in range(len(predictions_denorm)):
            pred_flat = predictions_denorm[i].flatten()
            target_flat = targets_denorm[i].flatten()
            
            diff_norm = np.sqrt(np.sum((pred_flat - target_flat) ** 2))
            target_norm = np.sqrt(np.sum(target_flat ** 2))
            
            rel_err = diff_norm / (target_norm + 1e-8)
            sample_errors.append(rel_err)
        
        sample_errors = np.array(sample_errors)
        
        # Результаты
        results = {
            'predictions': predictions_denorm,
            'targets': targets_denorm,
            'inputs': inputs,
            'metrics': {
                'rel_l2_normalized': rel_l2_norm,
                'rel_l2_physical': rel_l2_phys,
                'mse': mse,
                'max_error': max_err,
                'mean_sample_error': float(sample_errors.mean()),
                'std_sample_error': float(sample_errors.std()),
                'median_sample_error': float(np.median(sample_errors)),
                'min_sample_error': float(sample_errors.min()),
                'max_sample_error': float(sample_errors.max())
            },
            'sample_errors': sample_errors
        }
        
        # Вывод результатов
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(f"Относительная L2 ошибка (нормализованная): {rel_l2_norm:.6f}")
        print(f"Относительная L2 ошибка (физическая):     {rel_l2_phys:.6f}")
        print(f"MSE:                                       {mse:.6e}")
        print(f"Max error:                                 {max_err:.6e}")
        print(f"\nСтатистика по образцам:")
        print(f"  Mean error:   {sample_errors.mean():.6f}")
        print(f"  Std error:    {sample_errors.std():.6f}")
        print(f"  Median error: {np.median(sample_errors):.6f}")
        print(f"  Min error:    {sample_errors.min():.6f}")
        print(f"  Max error:    {sample_errors.max():.6f}")
        
        # Распределение ошибок
        print(f"\nПроцентили ошибок:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(sample_errors, p)
            print(f"  {p}%: {val:.6f}")
        
        # Сохранение результатов
        self._save_results(results)
        
        return results
    
    def _save_results(self, results):
        """Сохранение результатов в файлы."""
        # Создаём директорию
        self.config.RESULTS_DIR.mkdir(exist_ok=True)
        
        # Сохраняем метрики в JSON
        metrics_path = self.config.RESULTS_DIR / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        print(f"\nМетрики сохранены в {metrics_path}")
        
        # Сохраняем predictions в .npz
        data_path = self.config.RESULTS_DIR / "test_predictions.npz"
        np.savez_compressed(
            data_path,
            predictions=results['predictions'],
            targets=results['targets'],
            inputs=results['inputs'],
            sample_errors=results['sample_errors']
        )
        
        print(f"Predictions сохранены в {data_path}")
    
    def analyze_worst_cases(self, results, n_worst=5):
        """
        Анализ худших случаев (наибольшие ошибки).
        
        Параметры:
            results: результаты из evaluate_all()
            n_worst: число худших случаев для анализа
        """
        print("\n" + "=" * 60)
        print(f"АНАЛИЗ {n_worst} ХУДШИХ СЛУЧАЕВ")
        print("=" * 60)
        
        sample_errors = results['sample_errors']
        worst_indices = np.argsort(sample_errors)[-n_worst:][::-1]
        
        for rank, idx in enumerate(worst_indices, 1):
            error = sample_errors[idx]
            pred = results['predictions'][idx]
            target = results['targets'][idx]
            
            # Статистика предсказания
            pred_min = pred.min()
            pred_max = pred.max()
            pred_mean = pred.mean()
            
            target_min = target.min()
            target_max = target.max()
            target_mean = target.mean()
            
            print(f"\n{rank}. Образец #{idx}")
            print(f"   Относительная ошибка: {error:.6f}")
            print(f"   Prediction:  min={pred_min:.6f}, max={pred_max:.6f}, mean={pred_mean:.6f}")
            print(f"   Target:      min={target_min:.6f}, max={target_max:.6f}, mean={target_mean:.6f}")
    
    def compare_with_fem_time(self):
        """
        Сравнение времени inference FNO vs FEM.
        
        Примечание: для честного сравнения нужно засечь время
        генерации одного образца в generate_data.py
        """
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ ВРЕМЕНИ INFERENCE")
        print("=" * 60)
        
        # FNO inference time
        import time
        
        # Warmup
        dummy_input = self.test_dataset.inputs[:1]
        for _ in range(5):
            _ = self.predict(self.params, dummy_input)
        
        # Измерение
        n_samples = 100
        start_time = time.time()
        for i in range(n_samples):
            _ = self.predict(self.params, self.test_dataset.inputs[i:i+1])
        fno_time = (time.time() - start_time) / n_samples
        
        print(f"FNO inference (средн.): {fno_time*1000:.2f} ms")
        print(f"FNO throughput:         {1/fno_time:.1f} samples/sec")
        
        print("\nДля сравнения с FEM:")
        print("  Запустите generate_data.py с измерением времени")
        print("  Типичное время FEM для 128×128 mesh: ~100-500 ms")
        print(f"  Ожидаемое ускорение FNO: ~{100/fno_time/1000:.0f}-{500/fno_time/1000:.0f}x")


if __name__ == "__main__":
    import sys
    
    # Создание evaluator
    checkpoint_path = None
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    evaluator = Evaluator(checkpoint_path, Config)
    
    # Evaluation
    results = evaluator.evaluate_all()
    
    # Анализ худших случаев
    evaluator.analyze_worst_cases(results, n_worst=5)
    
    # Сравнение времени
    evaluator.compare_with_fem_time()