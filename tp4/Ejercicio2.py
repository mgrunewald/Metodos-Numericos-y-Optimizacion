import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# Obtener la ruta completa del directorio actual del script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Cambiar el directorio de trabajo actual a la ubicación del script
os.chdir(current_dir)

# Cargar los datos con ajustes
data = pd.read_csv('retornos_historicos.csv', skiprows=1, index_col=0)

# Asegurarse de que los datos sean numéricos
returns = data.apply(lambda x: x.str.replace(',', '').astype(float)).values.T

# Escalar los datos
returns_scaled = (returns - returns.min(axis=0)) / (returns.max(axis=0) - returns.min(axis=0))

# Definir la matriz de covarianza y el vector de retornos promedio
cov_matrix = np.cov(returns_scaled)
avg_returns = np.mean(returns_scaled, axis=1)

# Restricciones para las variables de decisión
num_assets = len(avg_returns)
bounds = tuple((0, 1) for asset in range(num_assets))

# Función objetivo y restricciones
def objective_function(x, cov_matrix):
    return x @ cov_matrix @ x.T

def constraint_sum(x):
    return np.sum(x) - 1

# Restricción de riesgo mínimo (no lineal)
def constraint_risk(x, rmin, avg_returns):
    return avg_returns @ x - rmin

# Resolver el problema de optimización con SLSQP
def optimize_portfolio(rmin, cov_matrix, avg_returns):
    # Variables para registrar la información de la iteración
    iterations = []
    objective_values = []
    risk_values = []
    values = []

    # Función objetivo y restricciones
    def objective_function(x, cov_matrix):
        # Registrar la información en cada iteración
        iterations.append(len(iterations) + 1)
        objective_values.append(x @ cov_matrix @ x.T)
        risk_values.append(avg_returns @ x)
        values.append(x.copy())  # Guardar una copia de la distribución de activos actual

        return x @ cov_matrix @ x.T

    def constraint_sum(x):
        return np.sum(x) - 1

    def constraint_risk(x, rmin, avg_returns):
        return avg_returns @ x - rmin

    initial_guess = np.ones(num_assets) / num_assets  # Distribución inicial uniforme

    # Restricciones
    constraints = ({'type': 'eq', 'fun': constraint_sum},
                   {'type': 'ineq', 'fun': lambda x: constraint_risk(x, rmin, avg_returns)})

    # Optimización cuadrática convexa con SLSQP
    result = minimize(objective_function, initial_guess, args=(cov_matrix,), constraints=constraints, method='SLSQP', bounds=bounds, options={'disp': True})

    return result.x, result.fun, result, iterations, objective_values, risk_values, values

# Función para visualizar la convergencia
def risk_barplot(r_mins, risks):
    plt.bar(np.arange(len(r_mins)), risks, align='center', alpha=0.7)
    plt.xlabel('Índice de r_min')
    plt.ylabel('Riesgo')
    
    # Ajusta el tamaño de la fuente en el eje x
    plt.xticks(np.arange(len(r_mins)), [f'r_min {r}' for r in r_mins], fontsize=8, rotation=45, ha='right')
    
    plt.show()


# Ejemplo de uso
rmin_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
risks = []
optimal_weights_list = []

for rmin in rmin_values:
    optimal_portfolio_weights, _, result, _, _, risk_values, _ = optimize_portfolio(rmin, cov_matrix, avg_returns)
    risks.append(avg_returns @ optimal_portfolio_weights)
    optimal_weights_list.append(optimal_portfolio_weights)

# Utiliza la función risk_barplot para mostrar el riesgo en función de r_min
risk_barplot(rmin_values, risks)

# Muestra las distribuciones óptimas de activos para cada r_min
for i, weights in enumerate(optimal_weights_list):
    print(f'Distribución óptima de activos para r_min {rmin_values[i]}: {weights}')
