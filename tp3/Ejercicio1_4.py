import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

imagen_seleccionada = imread('img02.jpeg') #elegir cualquier imagen del data set

imagenes = ['img00.jpeg', 'img01.jpeg', 'img02.jpeg', 'img03.jpeg',
            'img04.jpeg', 'img05.jpeg', 'img06.jpeg', 'img07.jpeg',
            'img08.jpeg', 'img09.jpeg', 'img10.jpeg', 'img11.jpeg',
            'img12.jpeg', 'img13.jpeg', 'img14.jpeg', 'img15.jpeg']

vectores_imagenes = []
for imagen in imagenes:
    imagen = imread(imagen).astype(np.float64)

    if imagen is not None:
        vector_imagen = imagen.flatten()
        vectores_imagenes.append(vector_imagen)

# Convierte la lista de vectores en una matriz n x (p*p)
matriz_datos = np.array(vectores_imagenes)

U, S, Vt = np.linalg.svd(imagen_seleccionada, full_matrices=False)

# Error límite (10%) usando la norma de Frobenius
error_limite = 0.10 * np.linalg.norm(imagen_seleccionada, 'fro')

d = 0
error = np.inf
# Itera para encontrar el valor mínimo de d
for i in range(len(S)):
    matriz_reducida = np.dot(U[:, :i], np.dot(np.diag(S[:i]), Vt[:i, :]))
    error = np.linalg.norm(imagen_seleccionada - matriz_reducida, 'fro')

    if error <= error_limite:
        d = i
        break

Vt_truncado = Vt[:d, :] #(6,28)
v_truncado = Vt_truncado.T #(28,6)

representaciones_baja_dim = []
for imagen_original in matriz_datos:
    imagen_original = imagen_original.reshape((28, 28)) #(28,28)
    representacion_baja_dim = np.matmul(imagen_original, v_truncado) #(28,28)x(28,6)=(28,6)
    representacion_baja_dim = np.matmul(representacion_baja_dim, Vt_truncado) #(28,6)x(6,28)=(28x28)
    representacion_baja_dim = representacion_baja_dim.flatten()
    representaciones_baja_dim.append(representacion_baja_dim)

num_filas = 2
num_columnas = 8

fig, axs = plt.subplots(num_filas, num_columnas, figsize=(16, 4))

for i, representation in enumerate(representaciones_baja_dim):
    # Reshape de la representación aplanada a 28x28
    image = representation.reshape((28, 28))

    # Calcula la fila y columna actual del subplot
    fila_actual = i // num_columnas
    columna_actual = i % num_columnas

    # Asigna la imagen al subplot correspondiente
    axs[fila_actual, columna_actual].imshow(image, cmap='gray')
    axs[fila_actual, columna_actual].axis('off')

    # Agrega el número de imagen encima de cada subplot
    axs[fila_actual, columna_actual].set_title(f'Imagen {i}', fontsize=10)

# Ajusta el espaciado entre subplots
plt.subplots_adjust(wspace=0.1, hspace=0.4)

# Muestra las imágenes
plt.show()
