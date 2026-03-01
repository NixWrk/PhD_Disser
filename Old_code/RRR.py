import numpy as np
from scipy.special import legendre
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.special import eval_legendre

# Параметры модели (замени своими значениями, если нужно)
heart_a=90/1000
heart_b=45/1000

# # --- Геометрические параметры ---
h1 = 2e-2       # Глубина залегания неоднородности (м)

a0 = heart_a      # Половина расстояния между токовыми электродами (м)
b10 = heart_b     # Половина расстояния между потенциалными электродами (слева) (м)
br0 = b10     # Половина расстояния между потенциалными электродами (справа) (м)



# количество членов ряда
N = 100


rho1 = 9.228239e+00
rho2 = 1.350000e+00
Z_target =  43# целевое значение Z

# Сетка по X и Y
x_vals = np.linspace(-0.2, 0.2, 200)
y_vals = np.linspace(-0.2, 0.2, 200)
X, Y = np.meshgrid(x_vals, y_vals)

# Предварительные данные
n_values = np.arange(N + 1)
coefs = n_values * (rho2 - rho1) / ((n_values + 1) * rho2 + n_values * rho1)
P = [legendre(n) for n in n_values]

# Векторизованная функция Zbase как функция от r1
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
        return (rho1/(np.pi*r0)) * (series * Pn).sum(axis=0)

    dz_am = vectorized_component(RA, RM, cos_theta_AM)
    dz_an = vectorized_component(RA, RN, cos_theta_AN)
    dz_bm = vectorized_component(RB, RM, cos_theta_BM)
    dz_bn = vectorized_component(RB, RN, cos_theta_BN)

    dZ = dz_am - dz_an + dz_bn - dz_bm
    Zbase = (rho1 * 2 * br0) / (np.pi * (a0**2 - br0**2)) + dZ
    return Zbase

# Основная функция для нахождения r1(x, y)
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
                    R1[i, j] = np.nan  # Z_target вне допустимого диапазона
                    continue
                r1_sol = brentq(objective, 1e-4, 0.05)
                R1[i, j] = r1_sol
            except Exception:
                R1[i, j] = np.nan  # на случай других ошибок
    return R1


# Вычисляем поверхность r1(x, y)
#R1_surface = find_r1_surface(X, Y, Z_target)

# Визуализация
#fig = plt.figure(figsize=(10, 7))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, R1_surface, cmap='viridis')
#ax.set_title(f"Поверхность r1(x, y) при Z = {Z_target} м")
#ax.set_xlabel("X [м]")
#ax.set_ylabel("Y [м]")
#ax.set_zlabel("r1 [м]")
#plt.tight_layout()
#plt.show()



# Целевые параметры
Z_target_1 = 41.92
r1_value = 0.045  

Z_target_2 = 42.97
r2_value = 0.035  

# Вычисляем поверхности
print("Вычисляем R1_surface для Z_target_1...")
R1_surface_1 = find_r1_surface(X, Y, Z_target_1)

print("Вычисляем R1_surface для Z_target_2...")
R1_surface_2 = find_r1_surface(X, Y, Z_target_2)

# Функция для поиска точек пересечения с плоскостью
def find_intersection_points(R1_surface, target_r, X, Y, tol=1e-4):
    mask = np.abs(R1_surface - target_r) < tol
    return X[mask], Y[mask], R1_surface[mask]

# Поиск точек пересечения
x1, y1, z1 = find_intersection_points(R1_surface_1, r1_value, X, Y)
x2, y2, z2 = find_intersection_points(R1_surface_2, r2_value, X, Y)

# Визуализация точек пересечения
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, color='blue', label=f'Z={Z_target_1}, r1={r1_value} м', s=10)
ax.scatter(x2, y2, z2, color='red', label=f'Z={Z_target_2}, r2={r2_value} м', s=10)

ax.set_title("Точки пересечения поверхностей с плоскостями r1 и r2")
ax.set_xlabel("X [м]")
ax.set_ylabel("Y [м]")
ax.set_zlabel("r1 [м]")
ax.legend()
plt.tight_layout()
plt.show()