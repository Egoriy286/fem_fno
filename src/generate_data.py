"""
Генерация обучающих данных через DOLFINx.

Pipeline:
1. Генерация случайного коэффициента a(x) через Gaussian Random Field
2. Решение эллиптического PDE методом конечных элементов
3. Интерполяция решения на uniform grid для FNO
4. Сохранение датасета

Математическая постановка:
    -∇·(a(x)∇u(x)) = f(x)  в Ω = [0,1]²
    u = 0                   на ∂Ω

Слабая форма:
    ∫_Ω a(x) ∇u·∇v dx = ∫_Ω f(x) v dx,  ∀v ∈ H¹₀(Ω)
"""

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
import sys
from pathlib import Path
from tqdm import tqdm

from config import Config


class GaussianRandomField:
    """
    Генератор Gaussian Random Field для коэффициента a(x).
    
    Используем разложение по синусам и косинусам:
        a(x, y) = a₀ + σ Σ ξₖ φₖ(x, y)
    
    где ξₖ ~ N(0, 1), а φₖ — тригонометрические базисные функции.
    """
    
    def __init__(self, n_modes=12, sigma=0.8, base=2.0, length_scale=0.2):
        """
        Параметры:
            n_modes: число мод (чем больше, тем более сложная структура)
            sigma: амплитуда флуктуаций
            base: базовое значение a₀
            length_scale: характерный размер корреляций
        """
        self.n_modes = n_modes
        self.sigma = sigma
        self.base = base
        self.length_scale = length_scale
        
        # Генерируем частоты для базисных функций
        # Используем комбинации sin/cos с разными частотами
        self.frequencies = []
        for i in range(1, n_modes + 1):
            for j in range(1, n_modes + 1):
                # Затухание высоких частот
                weight = np.exp(-length_scale * (i**2 + j**2))
                self.frequencies.append((i, j, weight))
    
    def sample(self, x_coords, y_coords, rng):
        """
        Генерация одной реализации GRF.
        
        Параметры:
            x_coords: массив x-координат, shape (N,)
            y_coords: массив y-координат, shape (N,)
            rng: numpy random generator
        
        Возвращает:
            a_values: значения a(x,y) в точках, shape (N,)
        """
        # Инициализация базовым значением
        a_values = np.full_like(x_coords, self.base, dtype=np.float64)
        
        # Добавляем моды
        for k_x, k_y, weight in self.frequencies:
            # Случайные коэффициенты для sin и cos компонент
            c_sin = rng.normal(0, 1)
            c_cos = rng.normal(0, 1)
            
            # Добавляем вклад этой моды
            a_values += (self.sigma * weight * 
                        (c_sin * np.sin(2 * np.pi * k_x * x_coords) * np.sin(2 * np.pi * k_y * y_coords) +
                         c_cos * np.cos(2 * np.pi * k_x * x_coords) * np.cos(2 * np.pi * k_y * y_coords)))
        
        # Обеспечиваем положительность (эллиптичность)
        a_values = np.maximum(a_values, Config.A_MIN)
        
        return a_values


def create_mesh_and_function_space(comm):
    """
    Создание mesh и function space для FEM.
    
    Используем прямоугольную область [0,1]² с треугольными элементами.
    """
    # Создаём mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([Config.DOMAIN_X_MIN, Config.DOMAIN_Y_MIN]),
         np.array([Config.DOMAIN_X_MAX, Config.DOMAIN_Y_MAX])],
        [Config.FEM_NX, Config.FEM_NY],
        cell_type=mesh.CellType.triangle
    )
    
    # Function space: P1 элементы (линейные)
    V = fem.functionspace(domain, ("Lagrange", Config.FEM_DEGREE))
    
    return domain, V


def solve_elliptic_pde(V, a_field, f_value=1.0):
    """
    Решение эллиптического PDE с заданным коэффициентом a(x).
    
    Слабая форма:
        ∫ a(x) ∇u·∇v dx = ∫ f v dx
    
    Параметры:
        V: function space
        a_field: dolfinx.fem.Function, коэффициент a(x)
        f_value: значение правой части (скаляр или функция)
    
    Возвращает:
        u: решение (dolfinx.fem.Function)
    """
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Правая часть
    f = fem.Constant(V.mesh, PETSc.ScalarType(f_value))
    
    # Билинейная форма: a(x) ∇u·∇v
    a_form = ufl.inner(a_field * ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Линейная форма: f v
    L_form = f * v * ufl.dx
    
    # Граничные условия: u = 0 на ∂Ω
    # Находим все грани на границе
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Создаём DOF для граничных узлов
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Задаём нулевое значение
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Решение линейной системы
    problem = LinearProblem(
        a_form, L_form, 
        bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",  # прямой солвер
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
    )
    
    uh = problem.solve()
    
    return uh


def interpolate_to_uniform_grid(u_fem, domain, resolution):
    """
    Интерполяция FEM решения на uniform grid для FNO.
    
    FEM даёт решение на неструктурированной треугольной сетке.
    FNO требует данные на uniform grid (как изображение).
    
    Параметры:
        u_fem: dolfinx.fem.Function
        domain: mesh
        resolution: int, разрешение uniform grid (например, 64)
    
    Возвращает:
        u_grid: numpy array, shape (resolution, resolution)
        x_grid, y_grid: координаты uniform grid
    """
    # Создаём uniform grid
    x = np.linspace(Config.DOMAIN_X_MIN, Config.DOMAIN_X_MAX, resolution)
    y = np.linspace(Config.DOMAIN_Y_MIN, Config.DOMAIN_Y_MAX, resolution)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Flatten для интерполяции
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    
    # Интерполяция
    # DOLFINx может интерполировать в точках
    u_values = u_fem.eval(points, domain.comm)
    
    # Reshape обратно в 2D grid
    u_grid = u_values.reshape(resolution, resolution)
    
    return u_grid, xx, yy


def generate_coefficient_field(V, grf, rng):
    """
    Генерация коэффициента a(x) как FEM функции.
    
    Параметры:
        V: function space
        grf: GaussianRandomField
        rng: numpy random generator
    
    Возвращает:
        a_fem: dolfinx.fem.Function с коэффициентом a(x)
        a_grid: numpy array на uniform grid (для FNO)
    """
    # Создаём функцию для a
    a_fem = fem.Function(V)
    
    # Получаем координаты DOF
    # В DOLFINx: geometry.x содержит координаты всех точек mesh
    # Нам нужны координаты DOF function space
    dof_coords = V.tabulate_dof_coordinates()
    x_coords = dof_coords[:, 0]
    y_coords = dof_coords[:, 1]
    
    # Генерируем значения через GRF
    a_values = grf.sample(x_coords, y_coords, rng)
    
    # Присваиваем значения функции
    a_fem.x.array[:] = a_values
    
    # Также интерполируем на uniform grid для входа FNO
    a_grid, _, _ = interpolate_to_uniform_grid(a_fem, V.mesh, Config.FNO_RESOLUTION)
    
    return a_fem, a_grid


def generate_single_sample(V, domain, grf, rng, sample_idx):
    """
    Генерация одного образца (a, u).
    
    Возвращает:
        a_grid: input для FNO, shape (resolution, resolution)
        u_grid: target для FNO, shape (resolution, resolution)
    """
    # 1. Генерируем коэффициент a(x)
    a_fem, a_grid = generate_coefficient_field(V, grf, rng)
    
    # 2. Решаем PDE
    u_fem = solve_elliptic_pde(V, a_fem, f_value=Config.RHS_VALUE)
    
    # 3. Интерполируем решение на uniform grid
    u_grid, _, _ = interpolate_to_uniform_grid(u_fem, domain, Config.FNO_RESOLUTION)
    
    return a_grid, u_grid


def generate_dataset():
    """
    Генерация полного датасета.
    
    Сохраняет в файл:
        - a_train, u_train
        - a_val, u_val
        - a_test, u_test
    """
    # MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("=" * 60)
        print("ГЕНЕРАЦИЯ ДАТАСЕТА ДЛЯ FNO")
        print("=" * 60)
        Config.print_config()
        print("\nСоздание mesh и function space...")
    
    # Создаём mesh и function space
    domain, V = create_mesh_and_function_space(comm)
    
    if rank == 0:
        print(f"Mesh создан: {V.mesh.topology.index_map(2).size_local} ячеек")
        print(f"Function space: {V.dofmap.index_map.size_local} DOF\n")
    
    # Инициализация GRF
    grf = GaussianRandomField(
        n_modes=Config.GRF_N_MODES,
        sigma=Config.A_SIGMA,
        base=Config.A_BASE
    )
    
    # Random generator с фиксированным seed
    rng = np.random.default_rng(Config.SEED)
    
    # Массивы для данных
    # Только root процесс собирает данные
    if rank == 0:
        a_all = np.zeros((Config.N_TOTAL, Config.FNO_RESOLUTION, Config.FNO_RESOLUTION))
        u_all = np.zeros((Config.N_TOTAL, Config.FNO_RESOLUTION, Config.FNO_RESOLUTION))
        
        print(f"Генерация {Config.N_TOTAL} образцов...")
        print("Это может занять несколько минут...\n")
    
    # Генерация образцов
    for i in tqdm(range(Config.N_TOTAL), disable=(rank != 0)):
        a_grid, u_grid = generate_single_sample(V, domain, grf, rng, i)
        
        if rank == 0:
            a_all[i] = a_grid
            u_all[i] = u_grid
    
    # Сохранение (только root)
    if rank == 0:
        # Разделение на train/val/test
        a_train = a_all[:Config.N_TRAIN]
        u_train = u_all[:Config.N_TRAIN]
        
        a_val = a_all[Config.N_TRAIN:Config.N_TRAIN + Config.N_VAL]
        u_val = u_all[Config.N_TRAIN:Config.N_TRAIN + Config.N_VAL]
        
        a_test = a_all[Config.N_TRAIN + Config.N_VAL:]
        u_test = u_all[Config.N_TRAIN + Config.N_VAL:]
        
        # Проверка статистики
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ДАТАСЕТА")
        print("=" * 60)
        print(f"Коэффициент a(x):")
        print(f"  min:  {a_all.min():.4f}")
        print(f"  max:  {a_all.max():.4f}")
        print(f"  mean: {a_all.mean():.4f}")
        print(f"  std:  {a_all.std():.4f}")
        print(f"\nРешение u(x):")
        print(f"  min:  {u_all.min():.6f}")
        print(f"  max:  {u_all.max():.6f}")
        print(f"  mean: {u_all.mean():.6f}")
        print(f"  std:  {u_all.std():.6f}")
        
        # Сохранение
        Config.setup_directories()
        
        print(f"\nСохранение в {Config.DATA_FILE}...")
        np.savez_compressed(
            Config.DATA_FILE,
            a_train=a_train, u_train=u_train,
            a_val=a_val, u_val=u_val,
            a_test=a_test, u_test=u_test
        )
        
        print("Датасет успешно сгенерирован!")
        print("=" * 60)


if __name__ == "__main__":
    # Проверка аргументов командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        Config.apply_debug_mode()
    
    generate_dataset()