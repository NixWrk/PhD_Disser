import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import legendre


heart_a=90/1000
heart_b=45/1000

# # --- Геометрические параметры ---
h1 = 2e-2       # Глубина залегания неоднородности (м)
r1 = 4e-2       # Радиус неоднородности (м)
a0 = heart_a      # Половина расстояния между токовыми электродами (м)
b10 = heart_b     # Половина расстояния между потенциалными электродами (слева) (м)
br0 = b10     # Половина расстояния между потенциалными электродами (справа) (м)


# # --- Смещения центра неоднородности ---
x0 = 0       # Смещение по X (м)
y0 = 0      # Смещение по Y (м)

# количество членов ряда
N = 100


rho1 = 9.228239e+00
rho2 = 1.350000e+00


# Предварительно вычисляем коэффициенты для всех n
n_values = np.arange(N+1)
coefs = n_values * (rho2 - rho1) / ((n_values + 1)*rho2 + n_values*rho1)
# Предварительно генерируем полиномы Лежандра
P = [legendre(n) for n in n_values]

def calculate_Zbase(X, Y):
    # Векторизованные вычисления геометрических параметров
    x = X
    y = Y
    
    # Вычисляем все радиусы сразу для всех точек
    RA = np.sqrt((r1 + h1)**2 + (a0 - y)**2 + x**2)
    RB = np.sqrt((r1 + h1)**2 + (a0 + y)**2 + x**2)
    RM = np.sqrt((r1 + h1)**2 + (br0 - y)**2 + x**2)
    RN = np.sqrt((r1 + h1)**2 + (b10 + y)**2 + x**2)
    
    # Вычисляем косинусы углов для всех комбинаций
    cos_theta_AM = (RA**2 + RM**2 - (a0 - br0)**2) / (2 * RA * RM)
    cos_theta_AN = (RA**2 + RN**2 - (a0 + b10)**2) / (2 * RA * RN)
    cos_theta_BM = (RB**2 + RM**2 - (a0 + br0)**2) / (2 * RB * RM)
    cos_theta_BN = (RB**2 + RN**2 - (a0 - b10)**2) / (2 * RB * RN)
    
    # Вычисляем компоненты для всех точек сразу
    def vectorized_component(r0, r_in, cosθ):
        n = n_values[:, np.newaxis, np.newaxis]
        series = (r1/r_in)**(n+1) * (r1/r0)**n * coefs[:, np.newaxis, np.newaxis]
        Pn = np.array([p(cosθ) for p in P])
        return (rho1/(np.pi*r0)) * (series * Pn).sum(axis=0)
    
    # Вычисляем все компоненты
    dz_am = vectorized_component(RA, RM, cos_theta_AM)
    dz_an = vectorized_component(RA, RN, cos_theta_AN)
    dz_bm = vectorized_component(RB, RM, cos_theta_BM)
    dz_bn = vectorized_component(RB, RN, cos_theta_BN)
    
    # Суммируем результат
    dZ = dz_am - dz_an + dz_bn - dz_bm
    Zbase = (rho1 * 2 * br0) / (np.pi * (a0**2 - br0**2)) + dZ
    
    return Zbase

# Генерация сетки
x_vals = np.linspace(-0.02, 0.02, 100)
y_vals = np.linspace(-0.02, 0.02, 100)
X, Y = np.meshgrid(x_vals, y_vals)

Z_Zbase = calculate_Zbase(X, Y)


#Визуализация
fig = plt.figure(figsize=(10, 10))

# Zbase график
ax2 = fig.add_subplot(1, 1, 1, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_Zbase, cmap=cm.inferno)
ax2.set_title("Zbase(x, y)")
ax2.set_xlabel("x (м)")
ax2.set_ylabel("y (м)")
ax2.set_zlabel("Z (Ом)")

plt.tight_layout()
plt.show()



# Задаём диапазоны значений для r1 и h1
r1_vals = np.linspace(0.030, 0.060, 100)  # от 5 мм до 60 мм
h1_vals = np.linspace(0.01, 0.03, 100)  # от 5 мм до 60 мм

R1, H1 = np.meshgrid(r1_vals, h1_vals)
Z_values = np.zeros_like(R1)

x_fixed = 0.045
y_fixed = 0.0

# Вычисляем Zbase при фиксированных x, y и разных r1, h1
for i in range(R1.shape[0]):
    for j in range(R1.shape[1]):
        r1 = R1[i, j]
        h1 = H1[i, j]

        # Заменяем значения в глобальной области
        globals()['r1'] = r1
        globals()['h1'] = h1

        Z_values[i, j] = calculate_Zbase(np.array([[x_fixed]]), np.array([[y_fixed]]))[0, 0]

# Визуализация
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R1*1000, H1*1000, Z_values, cmap=cm.viridis)

ax.set_xlabel("Радиус r1 (мм)")
ax.set_ylabel("Глубина h1 (мм)")
ax.set_zlabel("Zbase (Ом)")
ax.set_title("Zbase(r1, h1) при фиксированных x=45мм, y=0")
plt.tight_layout()
plt.show()
