import numpy as np
import matplotlib.pyplot as plt

# Parámetros del péndulo
g = 9.81  # Aceleración debido a la gravedad (m/s^2)
l = 1.0   # Longitud de la cuerda (metros)
m = 1.0   # Masa del péndulo (kg)

# Parámetros de tiempo
t_inicio = 0
t_fin = 10
num_pasos = 1000

# Definir diferentes valores de ángulo inicial
theta0_values = [np.pi / 6, np.pi / 2]

# Crear una figura con subplots en una sola fila
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].set_ylabel('Angle (rad)')
axs[0].set_xlabel('Time')

# Funciones para calcular la energía cinética, potencial y total
def calcular_energia_cinetica(masa, longitud, velocidad_angular):
    return 0.5 * masa * (longitud * velocidad_angular)**2

def calcular_energia_potencial(masa, gravedad, longitud, angulo):
    return masa * gravedad * longitud * (1 - np.cos(angulo))

def calcular_energia_total(energia_cinetica, energia_potencial):
    return energia_cinetica + energia_potencial

# Método de Euler para resolver la ecuación del péndulo y calcular energías
for theta0 in theta0_values:
    h = (t_fin - t_inicio) / num_pasos  # Tamaño del paso de tiempo
    t_valores = [t_inicio]
    theta_valores = [theta0]
    omega_valores = [0.0]  # Inicializar velocidad angular con 0

    # Listas para almacenar las energías específicas para este valor de theta0
    energia_totalE = []
    energia_cineticaE = []
    energia_potencialE = []

    for i in range(num_pasos):
        t_nuevo = t_valores[-1] + h
        theta_nuevo = theta_valores[-1] + h * omega_valores[-1]
        omega_nuevo = omega_valores[-1] - (g / l) * np.sin(theta_valores[-1]) * h

        t_valores.append(t_nuevo)
        theta_valores.append(theta_nuevo)
        omega_valores.append(omega_nuevo)

        # Calcular la energía cinética y potencial en cada paso de tiempo
        energia_cinetica = calcular_energia_cinetica(m, l, l * omega_valores[-1])
        energia_potencial = calcular_energia_potencial(m, g, l, theta_valores[-1])
        energia_total = calcular_energia_total(energia_cinetica, energia_potencial)

        energia_cineticaE.append(energia_cinetica)
        energia_potencialE.append(energia_potencial)
        energia_totalE.append(energia_total)

    # Graficar la trayectoria en el primer subplot
    axs[0].plot(t_valores, theta_valores, label=f'θ0 = {theta0:.2f}')

    if theta0 == np.pi / 2:
        # En el segundo subplot, graficar la energía total junto con la cinética y potencial
        axs[1].plot(t_valores, [0] + energia_totalE, label=f'Total Energy θ0={theta0:.2f}')
        axs[1].plot(t_valores, [0] + energia_cineticaE, linestyle='--', label=f'Kinetic Energy θ0={theta0:.2f}')
        axs[1].plot(t_valores, [0] + energia_potencialE, linestyle='-.', label=f'Potential Energy θ0={theta0:.2f}')
        axs[1].set_ylabel('Energy Values')
        axs[1].set_xlabel('Time')

# Configurar los gráficos
for ax in axs:
    ax.grid(True)
    ax.legend()

# Ajustar el espaciado entre subplots
plt.tight_layout()
plt.show()

# Runge Kutta ------------------------------------------------------------------------------------------------------

# Función para calcular un paso de Runge-Kutta para el péndulo
def runge_kutta_step(theta, omega, h):
    k1_theta = h * omega
    k1_omega = h * (-g / l) * np.sin(theta)

    k2_theta = h * (omega + 0.5 * k1_omega)
    k2_omega = h * (-g / l) * np.sin(theta + 0.5 * k1_theta)

    k3_theta = h * (omega + 0.5 * k2_omega)
    k3_omega = h * (-g / l) * np.sin(theta + 0.5 * k2_theta)

    k4_theta = h * (omega + k3_omega)
    k4_omega = h * (-g / l) * np.sin(theta + k3_theta)

    theta_nuevo = theta + (1 / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_nuevo = omega + (1 / 6) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)

    return theta_nuevo, omega_nuevo

# Crear una figura para graficar las trayectorias
plt.figure(figsize=(8, 6))
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')

for theta0 in theta0_values:
    h = (t_fin - t_inicio) / num_pasos  # Tamaño del paso de tiempo
    t_valores = [t_inicio]
    theta_valores = [theta0]
    omega_valores = [0.0]  # Inicializar velocidad angular con 0

    for i in range(num_pasos):
        t_nuevo = t_valores[-1] + h
        theta_nuevo, omega_nuevo = runge_kutta_step(theta_valores[-1], omega_valores[-1], h)

        t_valores.append(t_nuevo)
        theta_valores.append(theta_nuevo)
        omega_valores.append(omega_nuevo)

    # Graficar la trayectoria
    if theta0 == theta0_values[0]:
        label = f'Theta Initial Angle {theta0:.2f}'
        color = 'green'
    else:
        label = f'Theta initial angle {theta0:.2f}'
        color = 'orange'
    plt.plot(t_valores, theta_valores, label=label, color=color)

plt.legend()
plt.grid(True)
plt.show()

# Grafico de las energías ---------------------------------------------------------------------------------

# Ángulo inicial más grande
theta0 = np.pi / 2  # Ángulo inicial de 90 grados

# Crear arrays para almacenar los valores de tiempo, ángulo, energía cinética y energía potencial
t_valores = [t_inicio]
theta_valores = [theta0]
omega_valores = [0.0]  # Inicializar velocidad angular con 0
energia_totalRk = []
energia_cineticaRk = []
energia_potencialRk = []

h = (t_fin - t_inicio) / num_pasos  # Tamaño del paso de tiempo

# Simulación utilizando Runge-Kutta
for i in range(num_pasos):
    t_nuevo = t_valores[-1] + h
    theta_nuevo, omega_nuevo = runge_kutta_step(theta_valores[-1], omega_valores[-1], h)
    t_valores.append(t_nuevo)
    theta_valores.append(theta_nuevo)
    omega_valores.append(omega_nuevo)

    # Calcular la energía cinética y potencial en cada paso de tiempo
    energia_cinetica = calcular_energia_cinetica(m, l, l * omega_valores[-1])
    energia_potencial = calcular_energia_potencial(m, g, l, theta_valores[-1])
    energia_total = calcular_energia_total(energia_cinetica, energia_potencial)

    energia_cineticaRk.append(energia_cinetica)
    energia_potencialRk.append(energia_potencial)
    energia_totalRk.append(energia_total)

# Asegurémonos de que ambas listas tengan la misma longitud
t_valores = t_valores[:-1]  # Elimina el último elemento de t_valores para que tengan la misma longitud

# Crear una figura para graficar la energía
plt.figure(figsize=(8, 6))
plt.xlabel('Time (s)')
plt.ylabel('Energy')

# Graficar la energía total junto con la cinética y potencial
plt.plot(t_valores, energia_totalRk, label='Total Energy')
plt.plot(t_valores, energia_cineticaRk, linestyle='--', label='Kinetic Energy')
plt.plot(t_valores, energia_potencialRk, linestyle='-.', label='Potential Energy')

plt.legend()
plt.grid(True)
plt.show()

# Gráficos de comparación de la resolución analítica y Runge-Kutta ---------------------------------------

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_ylabel('Angle (rad)')

omega0 = np.sqrt(g / l)

# Método de RK4 para resolver la ecuación del péndulo y calcular energías
for theta0 in theta0_values:
    h = (t_fin - t_inicio) / num_pasos  # Tamaño del paso de tiempo
    t_valores = [t_inicio]
    theta_valores = [theta0]
    omega_valores = [0.0]

    for i in range(num_pasos):
        theta_nuevo, omega_nuevo = runge_kutta_step(theta_valores[-1], omega_valores[-1], h)

        t_nuevo = t_valores[-1] + h

        t_valores.append(t_nuevo)
        theta_valores.append(theta_nuevo)
        omega_valores.append(omega_nuevo)

    # Graficar la trayectoria en el primer subplot
    ax1.plot(t_valores, theta_valores, label=f'RK4 θ0 = {theta0:.2f}')

    # Calcular y graficar la solución analítica
    t_analytical = np.linspace(t_inicio, t_fin, len(t_valores))
    theta_analytical = theta0 * np.cos(omega0 * t_analytical)
    ax1.plot(t_analytical, theta_analytical, linestyle='--', label=f'Analytical θ0={theta0:.2f}')

ax1.set_xlabel('Time (s)')
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()

# Error de la analítica con respecto a RK -----------------------------------------------------------------

total_energy_analiticaB = []
total_energy_rkB = []
t_valuesB = []
absolute_error = []

tita = (np.pi / 4)
v0 = 0
t0 = 0
t_max = 10.0
h = 0.01
omega_0 = np.sqrt(g / l)
tita0 = 0.1

while tita0 <= (np.pi / 2):
    tita_linearized = tita0 * np.cos(omega_0 * t0)
    tita0,v0=runge_kutta_step(tita0,omega_0,h)
    #afectar los titas por los metodos 

    # Calcular la energía total para Runge-Kutta
    energia_cinetica_rk = calcular_energia_cinetica(m, l, v0)
    energia_potencial_rk = calcular_energia_potencial(m, g, l, tita0)
    energiaTotal_rk = calcular_energia_total(energia_cinetica_rk, energia_potencial_rk)
    total_energy_rkB.append(energiaTotal_rk)

    # Calcular la energía total para la solución analítica
    energia_potencial_analitica = calcular_energia_potencial(m, g, l, tita_linearized)
    energia_cinetica_analitica = calcular_energia_cinetica(m, l, v0)
    energiaTotal_analitica = calcular_energia_total(energia_cinetica_analitica, energia_potencial_analitica)
    total_energy_analiticaB.append(energiaTotal_analitica)

    t0 += h
    t_valuesB.append(t0)

    # Calcular el error en este paso de tiempo
    error = np.abs(energiaTotal_rk - energiaTotal_analitica)
    absolute_error.append(error)

    tita0 += 0.1
    tita_linearized += 0.1

plt.figure(figsize=(10, 6))
plt.plot(t_valuesB, absolute_error, label="Abs Error")
plt.xlabel("Time")
plt.ylabel("Error in Energy")
plt.legend()
plt.grid(True)
plt.show()


#Error del step de h
# Define el rango de tamaños de paso (h) que deseas probar
h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Puedes ajustar esta lista según tus necesidades

# Inicializa una lista para almacenar los errores en la energía para cada valor de h
error_values = []

# Asegúrate de que todas las simulaciones tengan el mismo número de pasos
num_pasos_simulacion = len(total_energy_analiticaB)  # Igualamos el número de pasos al de la solución analítica

# Realiza simulaciones para cada valor de h
for h in h_values:
    # Inicializa listas para almacenar la energía total en cada paso de tiempo
    energia_total_simulacion = []
    
    # Inicializa el ángulo y la velocidad angular para cada simulación
    theta_simulacion = [np.pi / 4]
    omega_simulacion = [0.0]
    
    t_simulacion = 0.0
    
    for i in range(num_pasos_simulacion):
        # Realiza un paso de Runge-Kutta para el péndulo
        theta_nuevo, omega_nuevo = runge_kutta_step(theta_simulacion[-1], omega_simulacion[-1], h)
        t_simulacion += h
        
        # Calcula la energía total para este paso de tiempo
        energia_cinetica = calcular_energia_cinetica(m, l, l * omega_nuevo)
        energia_potencial = calcular_energia_potencial(m, g, l, theta_nuevo)
        energia_total = calcular_energia_total(energia_cinetica, energia_potencial)
        energia_total_simulacion.append(energia_total)
        
        # Actualiza el ángulo y la velocidad angular para el próximo paso
        theta_simulacion.append(theta_nuevo)
        omega_simulacion.append(omega_nuevo)
    
    # Calcula el error en la energía entre la simulación y la solución analítica
    error_simulacion = np.abs(energia_total_simulacion - np.array(total_energy_analiticaB))
    
    # Agrega el error al array de errores
    error_values.append(error_simulacion)

# Crea un gráfico de error en función del tamaño de paso h
plt.figure(figsize=(10, 6))
plt.plot(h_values, [error[0] for error in error_values], marker='o')

plt.xlabel("Step size (h)")
plt.ylabel("Energy Error")
plt.grid(True)
plt.show()

#diagrama de fases de rk

# Rango de ángulos iniciales y velocidades iniciales
theta0_range = np.linspace(-2 * np.pi, 2 * np.pi, 100)
omega0_range = np.linspace(-5, 5, 100)

# Crear una matriz para almacenar el espacio de fases
phase_space = np.zeros((len(theta0_range), len(omega0_range)))

# Calcular el espacio de fases
for i, theta0 in enumerate(theta0_range):
    for j, omega0 in enumerate(omega0_range):
        t_valores = [0.0]
        theta_valores = [theta0]
        omega_valores = [omega0]

        h = 0.1  # Tamaño del paso de tiempo
        num_pasos = 100  # Número de pasos

        for _ in range(num_pasos):
            t_nuevo = t_valores[-1] + h
            theta_nuevo, omega_nuevo = runge_kutta_step(theta_valores[-1], omega_valores[-1], h)
            t_valores.append(t_nuevo)
            theta_valores.append(theta_nuevo)
            omega_valores.append(omega_nuevo)

        # Almacena la velocidad angular en el espacio de fases
        phase_space[i, j] = omega_valores[-1]

# Graficar el espacio de fases
plt.figure(figsize=(8, 6))
plt.imshow(phase_space, extent=[omega0_range[0], omega0_range[-1], theta0_range[0], theta0_range[-1]], origin='lower', cmap='viridis')
plt.colorbar(label='Angular Velocity (rad/s)')
plt.xlabel('Initial Angular Velocity (rad/s)')
plt.ylabel('Initial Angle (rad)')
plt.grid(True)
plt.show()