# ==============================================================================
# #28DiasDePythonParaMineria - Día 26
# Título: Análisis de Incertidumbre Geológica con Simulación Condicional
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script va más allá de una simple estimación por Kriging, que solo da
# un resultado "suavizado" y promedio. Implementa una Simulación Gaussiana
# Secuencial (SGS) simplificada para generar múltiples "realizaciones" o
# escenarios posibles de la distribución de leyes que honran los datos de
# sondajes y el variograma. Esto permite cuantificar la incertidumbre
# geológica y generar mapas de probabilidad (P10, P50, P90).
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skgstat as skg
from scipy.spatial.distance import cdist
from tqdm import tqdm
import os

# --- 1. CARGA DE DATOS Y CONFIGURACIÓN GEOESTADÍSTICA ---
print("[1/5] Cargando datos y configurando el modelo de variograma...")
# Se espera un CSV con columnas 'x', 'y', 'grade'.
df = pd.read_csv('data/sample_points.csv')
# Normalizar los nombres de las columnas para evitar errores de mayúsculas/espacios.
df.columns = df.columns.str.strip().str.lower()
# Ajustar un modelo de variograma a los datos. Es la base de toda la geoestadística.
V = skg.Variogram(df[['x', 'y']].values, df['grade'].values,
                  model='spherical', n_lags=8, maxlag=100)

# --- 2. PREPARACIÓN DE LA GRILLA DE ESTIMACIÓN ---
print("[2/5] Creando la grilla para la estimación...")
grid_res = 5 # Resolución de la grilla (tamaño de bloque)
xx, yy = np.mgrid[0:100:grid_res, 0:100:grid_res]
grid_points = np.vstack((xx.ravel(), yy.ravel())).T

# --- 3. EJECUCIÓN DEL KRIGING (MANUAL Y ROBUSTO) ---
# Primero, realizamos una estimación por Kriging Ordinario en cada punto de la grilla.
# Esto nos da la estimación promedio (la media de la distribución condicional).
print("[3/5] Ejecutando Kriging para estimar la media y la varianza en cada punto...")
N_NEIGHBORS = 10 # Número de vecinos a usar en cada estimación.
kriging_field = np.zeros(len(grid_points))
kriging_variance = np.zeros(len(grid_points))
variogram_params = V.describe() # Extraer los parámetros del modelo (rango, sill, etc.)

def spherical_model(h, r, c0, b=0):
    """Implementación manual del modelo de variograma esférico."""
    a = r
    return np.piecewise(h, [h <= a, h > a], [
        lambda x: b + c0 * (1.5 * (x / a) - 0.5 * ((x / a) ** 3.0)),
        b + c0
    ])

# Bucle para estimar cada punto de la grilla. tqdm nos da una barra de progreso.
for i in tqdm(range(len(grid_points)), desc="Kriging Grid"):
    point = grid_points[i]
    
    # Encontrar los N vecinos más cercanos manualmente.
    distances = cdist([point], df[['x', 'y']].values)[0]
    n_max = min(N_NEIGHBORS, len(df))
    neighbor_indices = np.argsort(distances)[:n_max]
    neighbors = df.iloc[neighbor_indices]
    
    # Resolver el sistema de Kriging Ordinario para obtener los pesos.
    n = len(neighbors)
    coords = neighbors[['x', 'y']].values
    r, c0, b = variogram_params['effective_range'], variogram_params['sill'] - variogram_params['nugget'], variogram_params['nugget']
    
    A = np.ones((n + 1, n + 1))
    dist_matrix = cdist(coords, coords)
    A[:n, :n] = (c0 + b) - spherical_model(dist_matrix, r, c0, b)
    A[n, n] = 0
    
    b_vec = np.ones(n + 1)
    dist_vector = cdist([point], coords)[0]
    b_vec[:n] = (c0 + b) - spherical_model(dist_vector, r, c0, b)
    
    try:
        weights, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        # Calcular la estimación (media) y la varianza de Kriging.
        kriging_field[i] = np.dot(weights[:n], neighbors['grade'])
        kriging_variance[i] = np.dot(weights, b_vec)
    except np.linalg.LinAlgError:
        kriging_field[i] = np.nan
        kriging_variance[i] = np.nan

# Remodelar los arrays a la forma de la grilla.
kriging_field = kriging_field.reshape(xx.shape)
kriging_variance = kriging_variance.reshape(xx.shape)
kriging_std_dev = np.sqrt(np.maximum(0, kriging_variance)) # La desviación estándar es la raíz de la varianza.

# --- 4. GENERACIÓN DE REALIZACIONES CONDICIONALES ---
# Aquí es donde cuantificamos la incertidumbre.
print("[4/5] Generando realizaciones condicionales...")
N_REALIZATIONS = 100 # Número de escenarios posibles a generar.
# Generamos ruido gaussiano aleatorio.
noise = np.random.normal(0, 1, size=(N_REALIZATIONS,) + kriging_field.shape)
# Creamos las simulaciones. Cada simulación es: Media del Kriging + Ruido * Desviación Estándar del Kriging.
all_simulations = kriging_field + noise * kriging_std_dev

# --- 5. VISUALIZACIÓN ---
print("[5/5] Creando las visualizaciones finales...")
if not os.path.exists('output'): os.makedirs('output')

# Calcular los mapas de percentiles a partir de todas las realizaciones.
p10_map = np.percentile(all_simulations, 10, axis=0) # Caso optimista
p50_map = np.percentile(all_simulations, 50, axis=0) # Mediana (similar al Kriging)
p90_map = np.percentile(all_simulations, 90, axis=0) # Caso pesimista (nota: P90 es el percentil 90, no el valor que tiene 90% de prob de ser superado)

# --- Gráfico Estático (Panel de Mapas) ---
fig_static, axs = plt.subplots(2, 2, figsize=(14, 14))
fig_static.suptitle('Análisis de Incertidumbre Geológica (Simulación)', fontsize=20, weight='bold')
maps = {'Estimación (Kriging)': kriging_field.T, 'P10 (Optimista)': p10_map.T,
        'P50 (Mediana)': p50_map.T, 'P90 (Pesimista)': p90_map.T}
# Usar una escala de color consistente para todos los mapas.
vmin, vmax = np.nanmin(p90_map), np.nanmax(p10_map)
for ax, (title, data) in zip(axs.flatten(), maps.items()):
    im = ax.imshow(data, origin='lower', extent=[0, 100, 0, 100], cmap='viridis', vmin=vmin, vmax=vmax)
    ax.scatter(df['x'], df['y'], s=50, c='red', ec='black')
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
fig_static.colorbar(im, ax=axs, orientation='vertical', label='Ley (%)', shrink=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('output/uncertainty_maps.png', dpi=150)
plt.close(fig_static)

# --- GIF Animado (Carrusel de Realizaciones) ---
fig_anim, ax_anim = plt.subplots(figsize=(10, 10))
ax_anim.set_title("Carrusel de la Incertidumbre", fontsize=16)
ax_anim.set_xlabel('X'); ax_anim.set_ylabel('Y')
im_anim = ax_anim.imshow(all_simulations[0].T, origin='lower', extent=[0, 100, 0, 100],
                         cmap='viridis', vmin=vmin, vmax=vmax)
ax_anim.scatter(df['x'], df['y'], s=80, c='red', ec='black')
cbar_anim = fig_anim.colorbar(im_anim, label='Ley Simulada (%)')
title_anim = ax_anim.text(0.5, 1.02, '', ha='center', va='bottom', transform=ax_anim.transAxes, fontsize=14)
def animate(i):
    im_anim.set_data(all_simulations[i].T)
    title_anim.set_text(f'Realización: {i + 1}/{N_REALIZATIONS}')
    return [im_anim, title_anim]
ani = animation.FuncAnimation(fig_anim, animate, frames=N_REALIZATIONS, interval=100, blit=True)
ani.save('output/uncertainty_carousel.gif', writer='pillow', fps=10)
plt.close(fig_anim)

print("\n¡Éxito! Visualizaciones de incertidumbre generadas.")