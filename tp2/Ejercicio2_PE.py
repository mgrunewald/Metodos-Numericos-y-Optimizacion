import matplotlib.pyplot as plt
import numpy as np

# Definimos la ecuación diferencial (modelo logístico)
def population_growth(t, N, r=0.1, K=100, A=50):
    dNdt = r * N * (1 - N / K) * (N / A - 1)
    return dNdt

# Definición de rungeKutta
def rungeKutta(x0, y0, x, h, r):
    n = int((x - x0) / h)
    y = y0
    for i in range(1, n + 1):
        k1 = h * population_growth(x0, y, r)
        k2 = h * population_growth(x0 + 0.5 * h, y + 0.5 * k1, r)
        k3 = h * population_growth(x0 + 0.5 * h, y + 0.5 * k2, r)
        k4 = h * population_growth(x0 + h, y + k3, r)
        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x0 = x0 + h
    return y

# Parámetros de la simulación
a = 0  # Tiempo inicial
b = 100  # Tiempo final
N0 = 50  # Valor inicial de la población
h = 0.1  # Tamaño del paso para Runge-Kutta (ajusta según sea necesario)

# Listas para almacenar los valores de r y los tiempos estacionarios
r_values = np.linspace(0.01, 1, 20)
stationary_times = []

# Resolución de la ecuación con diferentes valores de r
for r in r_values:
    t = a
    N = N0
    while t <= b:
        previous_N = N
        N = rungeKutta(a, N0, t, h, r)
        if abs(N - previous_N) < 1e-5:
            stationary_times.append(t)
            break
        t += h

# Configura un gráfico para mostrar los tiempos estacionarios en función de los valores de r
plt.figure(figsize=(8, 6))
plt.plot(r_values, stationary_times)
plt.xlabel('r Values')
plt.ylabel('Stationary Time (t)')
plt.grid(True)
plt.title('Stationary Time vs. r')
plt.show()
