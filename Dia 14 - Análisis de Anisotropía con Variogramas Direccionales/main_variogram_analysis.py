# ==============================================================================
# #28DiasDePythonParaMineria - Día 14
# Título: Análisis de Anisotropía con Variogramas Direccionales
# Autor: Maycol Benavides
# The Bull Miner
# Descripción:
# Este script realiza un análisis geoestadístico para investigar la anisotropía
# en un conjunto de datos espaciales. Calcula y visualiza variogramas experimentales
# en diferentes direcciones para determinar si la continuidad espacial de una
# variable (como la ley de un mineral) cambia con la dirección.
#
# Técnica Clave:
# - Variograma: Una función que describe el grado de dependencia espacial de
#   un campo aleatorio o proceso estocástico. Es una herramienta fundamental
#   para el análisis de datos espaciales.
# - Anisotropía: La condición en la que las propiedades (en este caso, la
#   continuidad espacial) varían con la dirección.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg # scikit-gstat es una librería especializada en geoestadística

# --- 1. CARGA DE DATOS Y CONFIGURACIÓN ---
# Se definen los parámetros iniciales y se cargan los datos.
print("[1/3] Cargando datos y configurando el análisis...")

# Cargar los datos desde un archivo CSV.
# Se espera un formato con columnas 'x', 'y' (coordenadas) y 'value' (la variable a analizar).
df = pd.read_csv('data/sample_data.csv')

# Diccionario de parámetros para el cálculo del variograma.
# Esto centraliza la configuración y facilita los ajustes.
PARAMS = {
    'n_lags': 15,          # Número de "bins" o intervalos de distancia a calcular.
    'maxlag': 100,         # Distancia máxima a considerar en el análisis.
    'model': 'spherical'   # Modelo teórico a ajustar (esférico, exponencial, gaussiano, etc.).
}

# Lista de las direcciones (azimut en grados) que queremos analizar.
# 0°=Norte-Sur, 45°=Noreste-Suroeste, 90°=Este-Oeste, 135°=Noroeste-Sureste.
DIRECTIONS = [0, 45, 90, 135]

# --- 2. CÁLCULO DE VARIOGRAMAS DIRECCIONALES ---
# Iteramos sobre cada dirección definida y calculamos su variograma experimental.
print("[2/3] Calculando variogramas para cada dirección...")

# Lista para almacenar los objetos Variograma calculados.
directional_variograms = []
for az in DIRECTIONS:
    # Crear una instancia de la clase Variogram de scikit-gstat.
    V = skg.Variogram(
        coordinates=df[['x', 'y']].values, # Coordenadas de los puntos.
        values=df['value'].values,         # Valores de la variable en cada punto.
        azimuth=az,                        # Dirección de análisis.
        tolerance=15,                      # Tolerancia angular (ej. +/- 15°).
        bandwidth='auto',                  # Ancho de la banda de búsqueda perpendicular a la dirección.
        **PARAMS                           # Desempaquetar los parámetros comunes definidos arriba.
    )
    # El objeto `V` ahora contiene el variograma experimental y el modelo ajustado.
    directional_variograms.append(V)
    print(f"  - Variograma para {az}° calculado.")

# --- 3. VISUALIZACIÓN DE RESULTADOS ---
# Creamos un panel de gráficos para comparar los resultados de cada dirección.
print("[3/3] Creando el panel de gráficos comparativos...")

# Crear una figura con una grilla de subplots de 2x2.
# `sharex=True` y `sharey=True` aseguran que todos los gráficos tengan la misma escala,
# lo cual es crucial para la comparación visual.
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
fig.suptitle('Análisis de Anisotropía con Variogramas Direccionales', fontsize=20, weight='bold')

# Usamos zip para iterar sobre los ejes del gráfico, los objetos variograma y las direcciones
# de forma simultánea. Esto es más limpio y nos da acceso a todas las variables necesarias.
for ax, v, az in zip(axes.flatten(), directional_variograms, DIRECTIONS):
    
    # El método `plot()` de scikit-gstat es una forma rápida de dibujar
    # los puntos del variograma experimental.
    v.plot(axes=ax, grid=False, show=False)
    
    # Personalizamos el título del subplot usando el azimut de la iteración actual.
    ax.set_title(f'Azimut: {az}° (N{az}E)', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # scikit-gstat ajusta automáticamente un modelo teórico a los datos experimentales.
    # Extraemos los parámetros clave de este modelo para anotarlos en el gráfico.
    sill = v.describe()['sill']       # La meseta o varianza total.
    range_ = v.describe()['effective_range'] # La distancia a la que la correlación se pierde.
    nugget = v.describe()['nugget']   # La discontinuidad en el origen.
    
    # Dibujamos líneas de referencia para visualizar estos parámetros.
    ax.axhline(sill, color='red', linestyle=':', label=f'Sill: {sill:.2f}')
    ax.axvline(range_, color='green', linestyle=':', label=f'Rango: {range_:.1f}m')
    ax.legend()

# Añadir etiquetas de ejes globales a la figura para no repetirlas en cada subplot.
fig.text(0.5, 0.04, 'Distancia de Separación (h)', ha='center', va='center', fontsize=14)
fig.text(0.06, 0.5, 'Semivarianza (γ)', ha='center', va='center', rotation='vertical', fontsize=14)

# Ajustar el layout para que el título y las etiquetas no se superpongan.
plt.tight_layout(rect=[0.08, 0.08, 1, 0.95])
# Guardar la figura final en alta resolución.
plt.savefig('output/directional_variograms.png', dpi=300)

print("\n¡Éxito! Gráfico de variogramas direccionales guardado en 'output/directional_variograms.png'.")