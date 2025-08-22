import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, lagrange
from scipy.optimize import newton

# Cargar el archivo CSV con las mediciones
data = pd.read_csv("mnyo_mediciones.csv", delimiter=' ', header=None)

# Extraer las coordenadas x e y de las mediciones
xi = data[0].values
yi = data[1].values

# Cargar el archivo CSV con el ground truth
ground_truth = pd.read_csv("mnyo_ground_truth.csv", delimiter=' ', header=None)

# Extraer las coordenadas x e y del ground truth
x_gt = ground_truth[0].values
y_gt = ground_truth[1].values

# Crear un array con los valores del tiempo
time_values = np.linspace(0, len(xi) - 1, 100)

# Interpolación cúbica (Spline cúbico) para x e y
cs_x = CubicSpline(np.arange(len(xi)), xi)
cs_y = CubicSpline(np.arange(len(yi)), yi)

# Interpolación quintica (Spline quintico) para x e y
qs_x = PchipInterpolator(np.arange(len(xi)), xi)
qs_y = PchipInterpolator(np.arange(len(yi)), yi)

# Interpolación de Lagrange para x e y
lagrange_x = lagrange(np.arange(len(xi)), xi)
lagrange_y = lagrange(np.arange(len(yi)), yi)

# Valores interpolados para x e y usando los splines cúbicos, quinticos y Lagrange
x_values_cs = cs_x(time_values)
y_values_cs = cs_y(time_values)
x_values_qs = qs_x(time_values)
y_values_qs = qs_y(time_values)
x_values_lagrange = lagrange_x(time_values)
y_values_lagrange = lagrange_y(time_values)

# Graficar los puntos interpolados con diferentes métodos, el ground truth y los puntos de mediciones
plt.figure(figsize=(20, 12))

# Trazar la recta vertical x=10 en el índice 10
plt.axvline(x=10, color='pink', linestyle='-', label='x=10')

# Trazar la recta 0.35 * x1 + x2 = 3.6
x1_values = np.linspace(0, 10, 100)  # Valores de x1 en el rango de tus datos
x2_values = 3.6 - 0.35 * x1_values  # Calcular los valores correspondientes de x2
plt.plot(x1_values, x2_values, label='0.35 * x1 + x2 = 3.6', linestyle='-', color='pink')

# Resto de tu código para graficar los puntos y las otras líneas
plt.plot(x_gt, y_gt, label='Ground Truth', linestyle='-', color='black')
plt.plot(x_values_cs, y_values_cs, label='Spline Cúbico', linestyle='--', color='#FF0040')
plt.plot(x_values_qs, y_values_qs, label='Spline Quintico', linestyle='--', color='#8E44AD')
plt.plot(x_values_lagrange, y_values_lagrange, label='Interpolación de Lagrange', linestyle='--', color='#5499C7')
plt.scatter(xi, yi, label='Puntos de Mediciones', color='blue', marker='o')
plt.xlabel('Coordenada x')
plt.ylabel('Coordenada y')
plt.title('Interpolación de Puntos en el Plano XY')

# Encontrar la intersección con la interpolación cúbica usando Newton-Raphson
def intersection_equation1(x):
    return cs_x(x) - (3.6 - 0.35 * x)

def intersection_equation2(x):
    return cs_x(x) - 10

def find_root_with_refinement(func, initial_guess, tol, maxiter):
    while maxiter > 0:
        try:
            root = newton(func, initial_guess, tol=tol, maxiter=1)
            return root
        except RuntimeError:
            initial_guess += 0.01  # Adjust the guess
            maxiter -= 1
    raise RuntimeError("Root not found within the maximum number of iterations.")


# Inicializa una lista para almacenar las raíces de la primera ecuación
roots1 = []

# Iterate over a range of initial guesses for the first equation
for guess1 in np.linspace(0, 7, 50):
    try:
        root = newton(intersection_equation1, guess1, tol=1e-6, maxiter=1000)
        roots1.append(root)
    except RuntimeError:
        pass  # Ignore RuntimeError if it fails to converge for a specific guess

# Inicializa una lista para almacenar las raíces de la segunda ecuación
roots2 = []

# Iterate over a range of initial guesses for the second equation
for guess2 in np.linspace(0, 7, 50):
    try:
        root = newton(intersection_equation2, guess2, tol=1e-6, maxiter=1000)
        roots2.append(root)
    except RuntimeError:
        pass  # Ignore RuntimeError if it fails to converge for a specific guess

# Graficar las raíces encontradas
plt.scatter(roots1, [cs_y(root) for root in roots1], label='Intersecciones', color='green', marker='x', s=100)
plt.scatter(roots2, [cs_y(root) for root in roots2], label='Intersecciones', color='green', marker='x', s=100)

plt.legend()
plt.grid(True)
plt.show()
plt.tight_layout()
plt.show()