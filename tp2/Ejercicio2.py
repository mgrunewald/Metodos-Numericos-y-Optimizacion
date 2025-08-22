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
N0_values = [5, 15, 35, 50, 65, 75, 90, 100, 125, 150]  # Valores iniciales de la población
h = 0.1  # Tamaño del paso para Runge-Kutta (ajusta según sea necesario)

# Lista para almacenar los tiempos estacionarios para diferentes valores de r
stationary_times = []

# Resolución de la ecuación con diferentes valores de r y N0
r_values = np.linspace(0.01, 0.5, 100)

for r in r_values:
    times = []  # Lista para almacenar los tiempos de equilibrio para un valor de r dado
    for N0 in N0_values:
        x = []  # Lista para almacenar los valores de tiempo
        previous_N = N0
        for t in range(int(a), int(b) + 1):
            N = rungeKutta(a, N0, t, h, r)
            if abs(N - previous_N) < 1e-5:
                times.append(t)
                break
            previous_N = N
    stationary_times.append(times)

# Configura un nuevo gráfico para mostrar los tiempos estacionarios en función de los valores de r
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# Gráfico de evolución del tamaño de la población para diferentes N0 usando Runge-Kutta
for i, N0 in enumerate(N0_values):
    plt.plot(range(int(a), int(b) + 1), [rungeKutta(a, N0, t, h, 0.1) for t in range(int(a), int(b) + 1)], label=f'N0 = {N0}')

plt.xlabel('Time')
plt.ylabel('Population size (N)')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(1, 2, 2)

# Gráfico de tiempos estacionarios en función de los valores de r
for i in range(len(N0_values)):
    plt.plot(r_values, [times[i] for times in stationary_times], label=f'N0 = {N0_values[i]}')

plt.xlabel('r Values')
plt.ylabel('Stationary Time (t)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
