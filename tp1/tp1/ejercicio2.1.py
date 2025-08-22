import numpy as np
from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

#definimos función 1

def f1(x):
    return 0.05 * abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2


#definimos función 2

def f2(x1,x2):
    term1 = 0.7 * math.exp(-((9*x1 - 2)*2)/4 - ((9*x2 - 2)*2)/4)
    term2 = 0.45 * math.exp(-((9*x1 + 1)*2)/9 - ((9*x2 + 1)*2)/5)
    term3 = 0.55 * math.exp(-((9*x1 - 6)*2)/4 - ((9*x2 - 3)*2)/4)
    term4 = -0.1 * math.exp(-((9*x1 - 7)*2)/4 - ((9*x2 - 3)*2)/4)
    
    return term1 + term2 + term3 + term4


#definimos los puntos de interpolación para f1
# Por ejemplo, 10 puntos equidistantes
interpolation_points_f1 = np.linspace(-3, 3, 10)  


#calculamos los valores reales de la función en los puntos a interpolar
real_values_f1 = f1(interpolation_points_f1)

# Realizar interpolación con polinomio de Lagrange
lagrange_poly_f1 = lagrange(interpolation_points_f1, real_values_f1)

# Realizar interpolación con spline cúbico
spline_cubic_f1 = CubicSpline(interpolation_points_f1, real_values_f1)

# Realizar interpolación con spline quíntuple (quintic spline)
spline_quintic_f1= PchipInterpolator(interpolation_points_f1, real_values_f1)


# Puntos donde se evaluarán los resultados
evaluation_points_f1 = np.linspace(-3, 3, 100)  # Más puntos para una representación suave


# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolated = lagrange_poly_f1(evaluation_points_f1)
spline_cubic_interpolated = spline_cubic_f1(evaluation_points_f1)
spline_quintic_interpolated = spline_quintic_f1(evaluation_points_f1)


# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points_f1, f1(evaluation_points_f1), color='purple', label='Función Original')
plt.plot(evaluation_points_f1, lagrange_interpolated, label='Interpolación Lagrange')
plt.plot(evaluation_points_f1, spline_cubic_interpolated, label='Spline Cúbico')
plt.plot(evaluation_points_f1, spline_quintic_interpolated, label='Spline Quintuple')
plt.scatter(interpolation_points_f1, real_values_f1, color='black', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.title('Comparación de Métodos de Interpolación')
plt.legend()
plt.grid(True)
plt.show()


#%%
#definimos los puntos de interpolación para f2

# Por ejemplo, 10 puntos equidistantes
interpolation_points_f2 = np.linspace(-1, 1, 250)  


#calculamos los valores reales de la función en los puntos a interpolar
real_values_f2 = np.array([f2(x1, x2) for x1, x2 in zip(interpolation_points_f2, interpolation_points_f2)])


# Realizar interpolación con polinomio de Lagrange

lagrange_poly_f2 = lagrange(interpolation_points_f2, real_values_f2)

# Realizar interpolación con spline cúbico
spline_cubic_f2 = CubicSpline(interpolation_points_f2, real_values_f2)

# Realizar interpolación con spline quíntuple (quintic spline)
spline_quintic_f2= PchipInterpolator(interpolation_points_f2, real_values_f2)


# Puntos donde se evaluarán los resultados
evaluation_points_f2 = np.linspace(-1, 1, 250)  # Más puntos para una representación suave


# Evaluar los métodos de interpolación en los puntos de evaluación
lagrange_interpolated_f2 = lagrange_poly_f2(evaluation_points_f2)
spline_cubic_interpolated_f2 = spline_cubic_f2(evaluation_points_f2)
spline_quintic_interpolated_f2 = spline_quintic_f2(evaluation_points_f2)


# Graficar resultados para f2
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points_f2, real_values_f2, color='purple', label='Función Original')
plt.plot(evaluation_points_f2, lagrange_interpolated_f2, label='Interpolación Lagrange')
plt.plot(evaluation_points_f2, spline_cubic_interpolated_f2, label='Spline Cúbico')
plt.plot(evaluation_points_f2, spline_quintic_interpolated_f2, label='Spline Quintuple')
plt.scatter(interpolation_points_f2, real_values_f2, color='black', label='Puntos de Interpolación')
plt.xlabel('x2')
plt.ylabel('f2(x1, x2)')
plt.title('Comparación de Métodos de Interpolación para f2')
plt.legend()
plt.grid(True)
plt.show()