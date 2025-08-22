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

# Calculate the absolute difference between original and interpolated values
lagrange_absolute_diff = np.abs(f1(evaluation_points_f1) - lagrange_interpolated)
spline_cubic_absolute_diff = np.abs(f1(evaluation_points_f1) - spline_cubic_interpolated)
spline_quintic_absolute_diff = np.abs(f1(evaluation_points_f1) - spline_quintic_interpolated)

# Calculate the relative errors
lagrange_relative_error = (lagrange_absolute_diff / f1(evaluation_points_f1)) * 100
spline_cubic_relative_error = (spline_cubic_absolute_diff / f1(evaluation_points_f1)) * 100
spline_quintic_relative_error = (spline_quintic_absolute_diff / f1(evaluation_points_f1)) * 100

# Print error information for f1
max_absolute_error_f1 = np.max(lagrange_absolute_diff)
max_absolute_error_location_f1 = evaluation_points_f1[np.argmax(lagrange_absolute_diff)]
print(f"Error máximo con respecto a f1 (Lagrange): {max_absolute_error_f1} at x = {max_absolute_error_location_f1}")

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points_f1, f1(evaluation_points_f1), color='purple', label='Función Original')
plt.plot(evaluation_points_f1, lagrange_interpolated, label='Interpolación Lagrange')
plt.plot(evaluation_points_f1, spline_cubic_interpolated, label='Spline Cúbico')
plt.plot(evaluation_points_f1, spline_quintic_interpolated, label='Spline Quintuple')
plt.scatter(interpolation_points_f1, real_values_f1, color='black', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.title('Comparación de Métodos de Interpolación con 10 puntos equidistantes')
plt.legend()
plt.grid(True)
plt.show()

# Ploteo del error relativo de las interpolaciones
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points_f1, lagrange_relative_error, label='Lagrange')
plt.plot(evaluation_points_f1, spline_cubic_relative_error, label='Spline Cúbico')
plt.plot(evaluation_points_f1, spline_quintic_relative_error, label='Spline Quintuple')
plt.xlabel('x')
plt.ylabel('Error relativo (%)')
plt.title('Error relativo de los métodos de interpolación con 10 puntos equidistantes')
plt.legend()
plt.grid(True)
plt.show()


#  Número de puntos para la interpolación
n_points = 10

# Generar puntos de Chebyshev en el intervalo [-1, 1]
chebyshev_points = np.cos(np.pi * (2 * np.arange(n_points) + 1) / (2 * n_points))

# Transformar los puntos de Chebyshev al interval [-3, 3]
interpolation_points = 3 * chebyshev_points

# Ordenar los puntos de interpolación en orden creciente
sorted_indices = np.argsort(interpolation_points)
interpolation_points_sorted = interpolation_points[sorted_indices]
real_values_sorted = f1(interpolation_points_sorted)

# Definir puntos de evaluación suaves
evaluation_points = np.linspace(-3, 3, 100)

# Interpolación con Lagrange
lagrange_poly = lagrange(interpolation_points_sorted, real_values_sorted)
lagrange_interpolated = lagrange_poly(evaluation_points)

# Interpolación con Splines Cúbicos
spline_cubic = CubicSpline(interpolation_points_sorted, real_values_sorted)
spline_cubic_interpolated = spline_cubic(evaluation_points)

# Interpolación con Splines Quinticos
spline_quintic = PchipInterpolator(interpolation_points_sorted, real_values_sorted)
spline_quintic_interpolated = spline_quintic(evaluation_points)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points, f1(evaluation_points), color='purple', label='Función Original')
plt.plot(evaluation_points, lagrange_interpolated, label='Interpolación Lagrange')
plt.plot(evaluation_points, spline_cubic_interpolated, label='Spline Cúbico')
plt.plot(evaluation_points, spline_quintic_interpolated, label='Spline Quintuple')
plt.scatter(interpolation_points_sorted, real_values_sorted, color='black', label='Puntos de Interpolación')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparación de Métodos de Interpolación con puntos no equidistantes generados por una secuencia de Chebyshev')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the absolute difference between original and interpolated values
lagrange_absolute_diff = np.abs(f1(evaluation_points) - lagrange_interpolated)
spline_cubic_absolute_diff = np.abs(f1(evaluation_points) - spline_cubic_interpolated)
spline_quintic_absolute_diff = np.abs(f1(evaluation_points) - spline_quintic_interpolated)

# Calculate the relative errors
lagrange_relative_error = (lagrange_absolute_diff / f1(evaluation_points)) * 100
spline_cubic_relative_error = (spline_cubic_absolute_diff / f1(evaluation_points)) * 100
spline_quintic_relative_error = (spline_quintic_absolute_diff / f1(evaluation_points)) * 100

# Print error information for f1 with Chebyshev interpolation
max_relative_error_lagrange = np.max(lagrange_relative_error)
max_relative_error_location_lagrange = evaluation_points[np.argmax(lagrange_relative_error)]
print(f"Error máximo relativo con interpolación de Lagrange y puntos no equidistantes Chebyshev: {max_relative_error_lagrange}%")
print(f"Ubicación del error máximo relativo: {max_relative_error_location_lagrange}")

max_relative_error_spline_cubic = np.max(spline_cubic_relative_error)
max_relative_error_location_spline_cubic = evaluation_points[np.argmax(spline_cubic_relative_error)]
print(f"Error máximo relativo con interpolación de Splines cúbicos y puntos no equidistantes Chebyshev: {max_relative_error_spline_cubic}%")
print(f"Ubicación del error máximo relativo: {max_relative_error_location_spline_cubic}")

max_relative_error_spline_quintic = np.max(spline_quintic_relative_error)
max_relative_error_location_spline_quintic = evaluation_points[np.argmax(spline_quintic_relative_error)]
print(f"Error máximo relativo con interpolación de Splines quinticos y puntos no equidistantes Chebyshev: {max_relative_error_spline_quintic}%")
print(f"Ubicación del error máximo relativo: {max_relative_error_location_spline_quintic}")

# Ploteo del error relativo de las interpolaciones con Chebyshev points
plt.figure(figsize=(10, 6))
plt.plot(evaluation_points, lagrange_relative_error, label='Lagrange')
plt.plot(evaluation_points, spline_cubic_relative_error, label='Spline Cúbico')
plt.plot(evaluation_points, spline_quintic_relative_error, label='Spline Quintuple')
plt.xlabel('x')
plt.ylabel('Error relativo (%)')
plt.title('Error relativo de los métodos de interpolación con 10 puntos Chebyshev')
plt.legend()
plt.grid(True)
plt.show()

#definimos función 2
def f2(x1,x2):
    term1 = 0.7 * np.exp(-((9*x1 - 2)*2)/4 - ((9*x2 - 2)*2)/4)
    term2 = 0.45 * np.exp(-((9*x1 + 1)*2)/9 - ((9*x2 + 1)*2)/5)
    term3 = 0.55 * np.exp(-((9*x1 - 6)*2)/4 - ((9*x2 - 3)*2)/4)
    term4 = -0.1 * np.exp(-((9*x1 - 7)*2)/4 - ((9*x2 - 3)*2)/4)
    
    return term1 + term2 + term3 + term4

# Generar puntos de datos en el dominio bidimensional
data_points_x1 = np.linspace(-1, 1, 10)
data_points_x2 = np.linspace(-1, 1, 10)
data_points = np.array([(x1, x2) for x1 in data_points_x1 for x2 in data_points_x2])
data_values = np.array([f2(x1, x2) for x1, x2 in data_points])

# Crear una malla 3D de puntos de evaluación
xi, yi = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))

# Interpolamos con griddata
zi_interpolated = griddata(data_points, data_values, (xi, yi), method='cubic')

# Creamos la figura en 3d
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficamos la función original y la interpolada
ax.scatter(data_points[:, 0], data_points[:, 1], data_values, marker='o', label='Puntos de Datos')
surf = ax.plot_surface(xi, yi, zi_interpolated, cmap='plasma', alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f2(x1, x2)')
ax.set_title('Interpolación cúbica con griddata  de Función f2 en 3D con 10 puntos equidistantes')

# Agregamos una colorbar
colorbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
colorbar.set_label('Valor de Z')

# Crear una leyenda
plt.legend()

plt.show()

# Calculate the difference between the ground truth and the interpolated values
absolute_diff_f2 = np.abs(f2(xi, yi) - zi_interpolated)

# Calculate the relative error as a percentage
relative_error_f2 = (absolute_diff_f2 / f2(xi, yi)) * 100

# Print the maximum relative error and its location
max_relative_error_f2 = np.max(relative_error_f2)
max_relative_error_location_f2 = np.unravel_index(np.argmax(relative_error_f2), relative_error_f2.shape)
print(f"Máximo error relativo para f2 con 10 puntos equidistantes: {max_relative_error_f2}%")
print(f"Ubicación del máximo error relativo para f2: (x1={xi[max_relative_error_location_f2[0], max_relative_error_location_f2[1]]}, x2={yi[max_relative_error_location_f2[0], max_relative_error_location_f2[1]]})")

# Creamos la figura para visualizar el error
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Ploteamos el error
surf = ax.plot_surface(xi, yi, relative_error_f2, cmap='plasma', alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Error relativo (%)')
ax.set_title('Error relativo para la interpolación cúbica en 2d con griddata y 10 puntos equidistantes')

# Agregamos una Colorbar
colorbar = plt.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
colorbar.set_label('Error relativo (%)')

plt.show()


# Número de puntos para la interpolación en cada dimensión
n_points_x1 = 10
n_points_x2 = 10

# Generar puntos de Chebyshev en el intervalo [-1, 1] para ambas dimensiones
chebyshev_points_x1 = np.cos(np.pi * (2 * np.arange(n_points_x1) + 1) / (2 * n_points_x1))
chebyshev_points_x2 = np.cos(np.pi * (2 * np.arange(n_points_x2) + 1) / (2 * n_points_x2))

# Transformar los puntos de Chebyshev al intervalo [-1, 1]
interpolation_points_x1 = 2 * chebyshev_points_x1
interpolation_points_x2 = 2 * chebyshev_points_x2

# Crear una malla 2D de puntos de interpolación
data_points_x1, data_points_x2 = np.meshgrid(interpolation_points_x1, interpolation_points_x2)
data_points = np.array([(x1, x2) for x1, x2 in zip(data_points_x1.ravel(), data_points_x2.ravel())])
data_values = np.array([f2(x1, x2) for x1, x2 in data_points])

# Crear una figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Aumentar la densidad de la malla de evaluación
xi, yi = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))

# Interpolamos con griddata
zi_interpolated = griddata(data_points, data_values, (xi, yi), method='cubic')

# Plotear la superficie interpolada con una malla más densa
surf = ax.plot_surface(xi, yi, zi_interpolated, cmap='plasma', alpha=0.5)

# Plotear los puntos Chebyshev
ax.scatter(data_points_x1, data_points_x2, data_values, color='red', marker='o', label='Puntos de Chebyshev')

# Configurar ejes y etiquetas
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f2(x1, x2)')
ax.set_title('Interpolación cúbica con griddata de la función f2 con 10 puntos no equidistantes Chebyshev en 3D')

# Agregar una leyenda
ax.legend()

plt.show()

#diferencia absoluta
absolute_diff_f2 = np.abs(f2(xi, yi) - zi_interpolated)
#error relativo
relative_error_f2 = (absolute_diff_f2 / f2(xi, yi)) * 100
#maximo error y su ubicación 
max_relative_error_f2 = np.max(relative_error_f2)
max_relative_error_location_f2 = np.unravel_index(np.argmax(relative_error_f2), relative_error_f2.shape)

print(f"Máximo error relativo para f2 con puntos Chebyshev: {max_relative_error_f2}%")
print(f"Ubicación del máximo error relativo para f2 con puntos Chebyshev: (x1={xi[max_relative_error_location_f2[0], max_relative_error_location_f2[1]]}, x2={yi[max_relative_error_location_f2[0], max_relative_error_location_f2[1]]})")


# Crear una figura 3D para visualizar el error relativo
fig_error = plt.figure(figsize=(10, 8))
ax_error = fig_error.add_subplot(111, projection='3d')

# Plotear la superficie del error relativo
surf_error = ax_error.plot_surface(xi, yi, relative_error_f2, cmap='plasma', alpha=0.5)

# Configurar ejes y etiquetas
ax_error.set_xlabel('x1')
ax_error.set_ylabel('x2')
ax_error.set_zlabel('Error relativo (%)')
ax_error.set_title('Error relativo para la interpolación en 2D con puntos Chebyshev en 3D')

# Agregar una colorbar
colorbar_error = plt.colorbar(surf_error, ax=ax_error, shrink=0.6, aspect=10)
colorbar_error.set_label('Error relativo (%)')

plt.show()