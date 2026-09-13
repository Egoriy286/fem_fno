import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config


def relative_l2_error(pred, target):
    """
    Относительная L2 ошибка:
        ||pred - target||_2 / ||target||_2
    """
    return np.linalg.norm(pred - target) / (
        np.linalg.norm(target) + 1e-12
    )


def visualize_sample(
    data_path=None,
    sample_idx=6,
    save=True,
):
    """
    Визуализация одного тестового образца.

    Показывает:
        1. coefficient a(x,y)
        2. FEM solution
        3. FNO prediction
        4. absolute error
    """

    if data_path is None:
        data_path = Config.RESULTS_DIR / "test_predictions.npz"

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {data_path}"
        )

    print(f"Загрузка: {data_path}")

    data = np.load(data_path)

    inputs = data["inputs"]
    predictions = data["predictions"]
    targets = data["targets"]

    print(f"Inputs:       {inputs.shape}")
    print(f"Predictions:  {predictions.shape}")
    print(f"Targets:      {targets.shape}")

    if sample_idx >= len(predictions):
        raise IndexError(
            f"sample_idx={sample_idx}, "
            f"но доступно только {len(predictions)} образцов"
        )

    # ---------------------------------------------------------
    # Извлекаем sample
    # ---------------------------------------------------------

    sample_input = inputs[sample_idx]
    prediction = predictions[sample_idx]
    target = targets[sample_idx]

    # input shape: (H, W, C)
    # При FNO_USE_COORDS=True:
    #   channel 0 -> a(x,y)
    #   channel 1 -> x
    #   channel 2 -> y
    #
    # Если координаты не используются:
    #   channel 0 -> a(x,y)

    coefficient = sample_input[..., 0]

    # Убираем последний channel dimension
    prediction = np.squeeze(prediction)
    target = np.squeeze(target)

    # ---------------------------------------------------------
    # Ошибка
    # ---------------------------------------------------------

    error = np.abs(prediction - target)

    rel_l2 = relative_l2_error(
        prediction,
        target
    )

    max_error = error.max()
    mse = np.mean((prediction - target) ** 2)

    # ---------------------------------------------------------
    # Информация
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print(f"SAMPLE #{sample_idx}")
    print("=" * 60)

    print(f"Relative L2 error: {rel_l2:.6f}")
    print(f"Relative L2 error: {rel_l2 * 100:.3f}%")
    print(f"MSE:               {mse:.6e}")
    print(f"Max absolute error:{max_error:.6e}")

    print()
    print("Coefficient:")
    print(f"  min  = {coefficient.min():.6f}")
    print(f"  max  = {coefficient.max():.6f}")
    print(f"  mean = {coefficient.mean():.6f}")

    print()
    print("FEM target:")
    print(f"  min  = {target.min():.6e}")
    print(f"  max  = {target.max():.6e}")
    print(f"  mean = {target.mean():.6e}")

    print()
    print("FNO prediction:")
    print(f"  min  = {prediction.min():.6e}")
    print(f"  max  = {prediction.max():.6e}")
    print(f"  mean = {prediction.mean():.6e}")

    # ---------------------------------------------------------
    # Coordinate system
    # ---------------------------------------------------------

    x = np.linspace(
        Config.DOMAIN_X_MIN,
        Config.DOMAIN_X_MAX,
        coefficient.shape[0]
    )

    y = np.linspace(
        Config.DOMAIN_Y_MIN,
        Config.DOMAIN_Y_MAX,
        coefficient.shape[1]
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10)
    )

    # ---------------------------------------------------------
    # 1. Coefficient
    # ---------------------------------------------------------

    im0 = axes[0, 0].imshow(
        coefficient.T,
        origin="lower",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max()
        ],
        aspect="equal"
    )

    axes[0, 0].set_title(
        r"Coefficient $a(x,y)$"
    )

    axes[0, 0].set_xlabel("$x$")
    axes[0, 0].set_ylabel("$y$")

    fig.colorbar(
        im0,
        ax=axes[0, 0],
        fraction=0.046,
        pad=0.04
    )

    # ---------------------------------------------------------
    # 2. FEM target
    # ---------------------------------------------------------

    im1 = axes[0, 1].imshow(
        target.T,
        origin="lower",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max()
        ],
        aspect="equal"
    )

    axes[0, 1].set_title(
        r"FEM solution $u_{\mathrm{FEM}}$"
    )

    axes[0, 1].set_xlabel("$x$")
    axes[0, 1].set_ylabel("$y$")

    fig.colorbar(
        im1,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04
    )

    # ---------------------------------------------------------
    # 3. FNO prediction
    # ---------------------------------------------------------

    im2 = axes[1, 0].imshow(
        prediction.T,
        origin="lower",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max()
        ],
        aspect="equal"
    )

    axes[1, 0].set_title(
        r"FNO prediction $u_{\mathrm{FNO}}$"
    )

    axes[1, 0].set_xlabel("$x$")
    axes[1, 0].set_ylabel("$y$")

    fig.colorbar(
        im2,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04
    )

    # ---------------------------------------------------------
    # 4. Absolute error
    # ---------------------------------------------------------

    im3 = axes[1, 1].imshow(
        error.T,
        origin="lower",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max()
        ],
        aspect="equal"
    )

    axes[1, 1].set_title(
        r"Absolute error $|u_{\mathrm{FNO}}-u_{\mathrm{FEM}}|$"
    )

    axes[1, 1].set_xlabel("$x$")
    axes[1, 1].set_ylabel("$y$")

    fig.colorbar(
        im3,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04
    )

    # ---------------------------------------------------------
    # Overall title
    # ---------------------------------------------------------

    fig.suptitle(
        f"FNO inference — sample #{sample_idx} "
        f"(relative $L^2$ error = {rel_l2 * 100:.2f}%)",
        fontsize=14
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if save:
        output_path = (
            Config.RESULTS_DIR
            / f"sample_{sample_idx}_visualization.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        print()
        print(f"Figure сохранена: {output_path}")

    plt.show()


if __name__ == "__main__":

    visualize_sample(
        sample_idx=6
    )