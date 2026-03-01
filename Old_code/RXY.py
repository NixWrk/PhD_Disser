#Код для поиска возможных значений r1 для диапазона значений Z 

import numpy as np
from scipy.special import legendre, eval_legendre
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Параметры модели
heart_a = 90 / 1000
heart_b = 45 / 1000
h1 = 2e-2
a0 = heart_a
b10 = heart_b
br0 = b10
N = 100
rho1 = 9.228239
rho2 = 1.35

# Сетка по X и Y
x_vals = np.linspace(-0.02, 0.02, 50)
y_vals = np.linspace(-0.02, 0.02, 50)
X, Y = np.meshgrid(x_vals, y_vals)

# Предварительные данные
n_values = np.arange(N + 1)
coefs = n_values * (rho2 - rho1) / ((n_values + 1) * rho2 + n_values * rho1)

def Zbase_grid_given_r1(r1_value, X, Y):
    RA = np.sqrt((r1_value + h1)**2 + (a0 - Y)**2 + X**2)
    RB = np.sqrt((r1_value + h1)**2 + (a0 + Y)**2 + X**2)
    RM = np.sqrt((r1_value + h1)**2 + (br0 - Y)**2 + X**2)
    RN = np.sqrt((r1_value + h1)**2 + (b10 + Y)**2 + X**2)

    cos_theta_AM = (RA**2 + RM**2 - (a0 - br0)**2) / (2 * RA * RM)
    cos_theta_AN = (RA**2 + RN**2 - (a0 + b10)**2) / (2 * RA * RN)
    cos_theta_BM = (RB**2 + RM**2 - (a0 + br0)**2) / (2 * RB * RM)
    cos_theta_BN = (RB**2 + RN**2 - (a0 - b10)**2) / (2 * RB * RN)

    def vectorized_component(r0, r_in, cosθ):
        n = n_values[:, np.newaxis, np.newaxis]
        series = (r1_value/r_in)**(n+1) * (r1_value/r0)**n * coefs[:, np.newaxis, np.newaxis]
        Pn = eval_legendre(n_values[:, np.newaxis, np.newaxis], cosθ)
        return (rho1 / (np.pi * r0)) * (series * Pn).sum(axis=0)

    dz_am = vectorized_component(RA, RM, cos_theta_AM)
    dz_an = vectorized_component(RA, RN, cos_theta_AN)
    dz_bm = vectorized_component(RB, RM, cos_theta_BM)
    dz_bn = vectorized_component(RB, RN, cos_theta_BN)

    dZ = dz_am - dz_an + dz_bn - dz_bm
    Zbase = (rho1 * 2 * br0) / (np.pi * (a0**2 - br0**2)) + dZ
    return Zbase

def find_r1_surface(X, Y, Z_target):
    R1 = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            def objective(r1_local):
                return Zbase_grid_given_r1(r1_local, X[i:i+1, j:j+1], Y[i:i+1, j:j+1])[0, 0] - Z_target
            try:
                z1 = objective(1e-4)
                z2 = objective(0.05)
                if z1 * z2 > 0:
                    continue
                r1_sol = brentq(objective, 1e-4, 0.05)
                R1[i, j] = r1_sol
            except Exception:
                continue
    return R1

# --- ЭТАП 1: Диапазон возможных Z ---
Z_min_grid = Zbase_grid_given_r1(0.06, X, Y)  # r1 макс => Z мин
Z_max_grid = Zbase_grid_given_r1(0.01, X, Y)  # r1 мин => Z макс

Z_min_possible = np.nanmin(Z_min_grid)
Z_max_possible = np.nanmax(Z_max_grid)

print(f"Диапазон возможных значений Z: от {Z_min_possible:.3f} до {Z_max_possible:.3f}")

# --- ЭТАП 2: Диапазоны r1 для граничных Z с шагом отступа и счётчиком ---
def find_r1_range_for_Z(Z_init, direction, X, Y, delta_Z=0.001, max_iter=10, manual_start_z=None, Z_limit=None):
    Z_val = Z_init
    for shift_count in range(max_iter):
        print(f"Проверка Z = {Z_val:.5f} (отступ #{shift_count})")
        r1_surface = find_r1_surface(X, Y, Z_val)
        valid_r1 = r1_surface[~np.isnan(r1_surface)]
        if valid_r1.size > 0:
            r1_min = np.min(valid_r1)
            r1_max = np.max(valid_r1)
            print(f"→ Найдено! Z = {Z_val:.5f}, r1 ∈ [{r1_min:.6f}, {r1_max:.6f}] после {shift_count} отступов\n")
            return Z_val, r1_min, r1_max, shift_count
        Z_val += delta_Z * direction

    print("✖ Не удалось найти допустимый диапазон r1 за допустимое число шагов")

    # --- Повторная попытка с ручной Z, если задана ---
    if manual_start_z is not None:
        print(f"⏩ Повторный поиск от Z = {manual_start_z}")
        Z_val = manual_start_z
        shift_count = 0
        while Z_limit is None or (Z_val * direction <= Z_limit * direction):
            print(f"Проверка Z = {Z_val:.5f} (ручной отступ #{shift_count})")
            r1_surface = find_r1_surface(X, Y, Z_val)
            valid_r1 = r1_surface[~np.isnan(r1_surface)]
            if valid_r1.size > 0:
                r1_min = np.min(valid_r1)
                r1_max = np.max(valid_r1)
                print(f"→ Найдено! Z = {Z_val:.5f}, r1 ∈ [{r1_min:.6f}, {r1_max:.6f}] после ручного отступа #{shift_count}\n")
                return Z_val, r1_min, r1_max, max_iter + shift_count
            Z_val += delta_Z * direction
            shift_count += 1

    return None, None, None, None

# Для верхнего предела Z
Z_max_final, r1_min_maxZ, r1_max_maxZ, shift_max = find_r1_range_for_Z(
    Z_max_possible, direction=-1, X=X, Y=Y, 
    manual_start_z=Z_max_possible, Z_limit=Z_min_possible  # Ручное назначение границы с которой пойдёт поиск диапазона возможных значений r1

)

# Для нижнего предела Z
Z_min_final, r1_min_minZ, r1_max_minZ, shift_min = find_r1_range_for_Z(
    Z_min_possible, direction=+1, X=X, Y=Y, 
    manual_start_z=35.3, Z_limit=Z_max_possible  # Ручное назначение границы с которой пойдёт поиск диапазона возможных значений r1
)
