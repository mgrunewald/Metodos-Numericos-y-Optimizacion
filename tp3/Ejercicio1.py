import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.metrics import euclidean_distances

# Lista de las imágenes
nombres_imagenes = ['img00.jpeg', 'img01.jpeg', 'img02.jpeg', 'img03.jpeg',
                    'img04.jpeg', 'img05.jpeg', 'img06.jpeg', 'img07.jpeg',
                    'img08.jpeg', 'img09.jpeg', 'img10.jpeg', 'img11.jpeg',
                    'img12.jpeg', 'img13.jpeg', 'img14.jpeg', 'img15.jpeg']

vectores_imagenes = []
for nombre_imagen in nombres_imagenes:
    imagen = imread(nombre_imagen)

    if imagen is not None:
        vector_imagen = imagen.flatten()
        vectores_imagenes.append(vector_imagen)

# Convierte la lista de vectores en una matriz n x (p*p)
matriz_datos = np.array(vectores_imagenes)


# Realiza la descomposición SVD
U, S, Vt = np.linalg.svd(matriz_datos, full_matrices=False)

# Decide el valor de k (número de componentes principales a retener)
k = 10  # Se ajusta este valor para cada ploteo que quiera hacer

# Reconstruye la matriz original utilizando las primeras k componentes principales
matriz_reducida = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

# Visualizar en forma matricial p×p las imágenes reconstruidas con las últimas d dimensiones
d = 16  # Se ajusta este valor para cada ploteo que quiera hacer

# Reconstruye las imágenes utilizando las últimas d dimensiones
matriz_reducida_ultimas_d = np.dot(U[:, -d:], np.dot(np.diag(S[-d:]), Vt[-d:, :]))

fig, axs = plt.subplots(2, 16, figsize=(16, 4))
for row in axs:
    for ax in row:
        ax.axis('off')
fig.text(0.5, 0.89, 'Imágenes originales', ha='center', fontsize=10)
for i in range(16):
    axs[0, i].imshow(matriz_datos[i].reshape(28, 28), cmap="gray")
fig.text(0.5, 0.47, f'Imágenes reconstruidas con k = {k}', ha='center', fontsize=10)
for i in range(16):
    axs[1, i].imshow(matriz_reducida[i].reshape(28, 28), cmap="gray")

fig_ultimas_d, axs_ultimas_d = plt.subplots(1, 16, figsize=(16, 2))
for ax in axs_ultimas_d:
    ax.axis('off')
fig_ultimas_d.text(0.5, 1, f'Imágenes reconstruidas con últimas {d} dimensiones', ha='center', fontsize=10)
for i in range(16):
    axs_ultimas_d[i].imshow(matriz_reducida_ultimas_d[i].reshape(28, 28), cmap="gray")

valores_d = [5, 10, 16] # Valores de d a considerar para el mapa de calor
matrices_similitud_d = []

# Calcula las matrices de similitud para diferentes valores de d
for d in valores_d:
    matriz_reducida_d = np.dot(U[:, -d:], np.dot(np.diag(S[-d:]), Vt[-d:, :]))

    # Calcula la matriz de similitud entre las imágenes originales y las reconstruidas
    matriz_similitud_d = euclidean_distances(matriz_datos, matriz_reducida_d)
    matrices_similitud_d.append(matriz_similitud_d)

# Crea un gráfico para cada matriz de similitud basada en d
for i, d in enumerate(valores_d):
    plt.figure(figsize=(6, 6))
    plt.title(f'Matriz de Similaridad para d = {d}')
    sns.heatmap(matrices_similitud_d[i], cmap='viridis', square=True, cbar=True)
    plt.show()

# Valores de k a considerar para el mapa de calor
valores_k = [5, 10, 16]  
matrices_similitud_k = []

# Calcula las matrices de similitud para diferentes valores de k
for k in valores_k:
    # Reconstruye la matriz original utilizando las primeras k componentes principales
    matriz_reducida_k = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

    # Calcula la matriz de similitud entre las imágenes originales y las reconstruidas
    matriz_similitud_k = euclidean_distances(matriz_datos, matriz_reducida_k)
    matrices_similitud_k.append(matriz_similitud_k)

# Crea un gráfico para cada matriz de similitud basada en k
for i, k in enumerate(valores_k):
    plt.figure(figsize=(6, 6))
    plt.title(f'Matriz de Similaridad para k = {k}')
    sns.heatmap(matrices_similitud_k[i], cmap='viridis', square=True, cbar=True)
    plt.show()

plt.show()
