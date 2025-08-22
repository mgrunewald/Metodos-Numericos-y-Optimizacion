import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline, PchipInterpolator
from mpl_toolkits.mplot3d import Axes3D


x = [0, 8.517645089584206630, 1.623720889996237204, 12.86924818108026614, 18.17616982401723646, 12.33339209402911152, 21.77622054412167074, 24.57602173690377612, 24.48679983957896056, 33.44102121252131354]
y = [0.4472135954999579277, 1.549193338482966809, 2.144761058952721733, 2.607680962081059484, 3.000000000000000000, 3.346640106136302251, 3.305107438709036671, 3.012316347995543708, 2.672650740565235772, 2.293005338317756436]
z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

plt.title('Gráfico 3D de la Función')

plt.show()
