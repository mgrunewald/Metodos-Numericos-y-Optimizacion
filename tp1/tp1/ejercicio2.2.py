import numpy as np
from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

#definimos función 1
def f1(x):
    return 0.05 * abs(x) * np.sin(5 * x) + np.tanh(2 * x) + 2

#definimos los puntos de interpolación para f1
interpolation_points_f1 = np.linspace(-3, 3, 10)  # Por ejemplo, 10 puntos equidistantes

#calculamos los valores reales de la función en los puntos a interpolar
real_values_f1 = f1(interpolation_points_f1)

# Interpolamos con Lagrange
lagrange_poly_f1 = lagrange(interpolation_points_f1, real_values_f1)

# Interpolamos con Splines Cúbicos
spline_cubic_f1 = CubicSpline(interpolation_points_f1, real_values_f1)

# Interpolamos con Splines Quínticos
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


#definimos función 2
def f2(x1,x2):
    term1 = 0.7 * math.exp(-((9*x1 - 2)*2)/4 - ((9*x2 - 2)*2)/4)
    term2 = 0.45 * math.exp(-((9*x1 + 1)*2)/9 - ((9*x2 + 1)*2)/5)
    term3 = 0.55 * math.exp(-((9*x1 - 6)*2)/4 - ((9*x2 - 3)*2)/4)
    term4 = -0.1 * math.exp(-((9*x1 - 7)*2)/4 - ((9*x2 - 3)*2)/4)
    
    return term1 + term2 + term3 + term4

# Generar puntos de datos en el dominio bidimensional
data_points_x1 = np.linspace(-1, 1, 20)
data_points_x2 = np.linspace(-1, 1, 20)
data_points = np.array([(x1, x2) for x1 in data_points_x1 for x2 in data_points_x2])
data_values = np.array([f2(x1, x2) for x1, x2 in data_points])

# Crear una malla 3D de puntos de evaluación
xi, yi = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))

# Interpolamos con griddata
zi_interpolated = griddata(data_points, data_values, (xi, yi), method='linear')

# Creamos la figura en 3d
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficamos la función original y la interpolada
ax.scatter(data_points[:, 0], data_points[:, 1], data_values, marker='o', label='Puntos de Datos')
surf = ax.plot_surface(xi, yi, zi_interpolated, cmap='plasma', alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f2(x1, x2)')
ax.set_title('Interpolación de Función f2 en 3D')

# Agregamos una colorbar
colorbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
colorbar.set_label('Valor de Z')

# Crear una leyenda
plt.legend()

plt.show()