# ==============================================================================
# #28DiasDePythonParaMineria - Día 20
# Título: c
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script desmitifica el proceso de Kriging Ordinario. Lee un conjunto de
# datos de muestras, ajusta un modelo de variograma y luego genera una
# animación que muestra, paso a paso, cómo se estima el valor en cada punto
# de una grilla: seleccionando los vecinos más cercanos, calculando sus pesos
# y realizando la estimación ponderada.
#
# Nota de Desarrollo:
# La lógica para calcular los pesos de Kriging se implementa manualmente
# para asegurar la robustez y evitar depender de métodos internos de la
# librería `scikit-gstat` que pueden cambiar entre versiones.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skgstat as skg
from scipy.spatial.distance import cdist

# --- 1. CARGA DE DATOS Y CONFIGURACIÓN GEOESTADÍSTICA ---
# El primer paso es definir la estructura de correlación espacial de nuestros datos.
print("[1/5] Cargando datos y configurando el modelo de variograma...")
df = pd.read_csv('data/drillhole_data.csv')

# Crear un objeto Variograma. `scikit-gstat` ajusta automáticamente un modelo
# teórico (en este caso, esférico) a los datos experimentales.
V = skg.Variogram(df[['x', 'y']].values, df['grade'].values,
                  model='spherical', n_lags=5, maxlag=100)

# --- 2. PREPARACIÓN DE LA GRILLA DE ESTIMACIÓN ---
# Creamos una grilla regular de puntos donde realizaremos las estimaciones.
print("[2/5] Creando la grilla para la estimación por Kriging...")
grid_res = 10  # La distancia entre puntos de la grilla
xx, yy = np.mgrid[0:100:grid_res, 0:100:grid_res]
grid_points = np.vstack((xx.ravel(), yy.ravel())).T

# --- 3. EJECUCIÓN DEL KRIGING (PARA EL MAPA ESTÁTICO) ---
# Realizamos una estimación completa en toda la grilla para tener un mapa de fondo.
print("[3/5] Ejecutando Kriging para generar el mapa de estimación final...")
N_NEIGHBORS = 5 # Definimos cuántos vecinos cercanos usará el algoritmo.
ok_map = skg.OrdinaryKriging(V, min_points=2, max_points=N_NEIGHBORS, mode='exact')
kriging_field = ok_map.transform(grid_points[:, 0], grid_points[:, 1]).reshape(xx.shape).T

# --- 4. VISUALIZACIÓN ESTÁTICA ---
# Guardamos una imagen del resultado final para el post.
print("[4/5] Creando la imagen estática del mapa de Kriging...")
fig_static, ax_static = plt.subplots(figsize=(8, 8))
vmin, vmax = np.nanmin(kriging_field), np.nanmax(kriging_field) # Para consistencia de colores
im = ax_static.imshow(kriging_field, origin='lower', extent=(0, 100, 0, 100), cmap='viridis', vmin=vmin, vmax=vmax)
ax_static.scatter(df['x'], df['y'], c=df['grade'], s=100, ec='black', cmap='viridis', vmin=vmin, vmax=vmax)
ax_static.set_title('Mapa de Estimación por Kriging Ordinario', fontsize=16)
ax_static.set_xlabel('Coordenada X'); ax_static.set_ylabel('Coordenada Y')
fig_static.colorbar(im, ax=ax_static, label='Ley Estimada (%)')
plt.savefig('output/kriging_map.png', dpi=150)
plt.close(fig_static)

# --- 5. CREACIÓN DEL GIF ANIMADO ---
print("[5/5] Creando el GIF animado del proceso de Kriging...")
fig_anim, ax_anim = plt.subplots(figsize=(10, 10))
# Para que la animación no sea eterna, solo animamos un subconjunto de los puntos de la grilla.
points_to_animate = grid_points[::10]

# --- Funciones de Ayuda para la Animación ---

def spherical_model(h, r, c0, b=0):
    """Implementación manual del modelo de variograma esférico."""
    a = r # El rango práctico 'a' es el parámetro de rango 'r'
    # np.piecewise es una forma eficiente de aplicar una condición if/else a un array.
    return np.piecewise(h, [h <= a, h > a], [
        lambda x: b + c0 * (1.5 * (x / a) - 0.5 * ((x / a) ** 3.0)), # Si h <= r
        b + c0                                                      # Si h > r
    ])

def solve_kriging_system(target_point, neighbors, variogram_params):
    """Resuelve el sistema de ecuaciones de Kriging Ordinario manualmente."""
    n = len(neighbors)
    coords = neighbors[['x', 'y']].values
    
    # Extraer los parámetros del variograma del diccionario.
    # 'effective_range' es la clave correcta que usa scikit-gstat.
    r = variogram_params['effective_range']
    sill = variogram_params['sill']
    nugget = variogram_params['nugget']
    c0 = sill - nugget # C0 es el sill parcial

    # Construir la matriz A (matriz de covarianza entre los vecinos)
    A = np.ones((n + 1, n + 1))
    dist_matrix = cdist(coords, coords)
    # Covarianza = Sill Total - Variograma(h)
    A[:n, :n] = (c0 + nugget) - spherical_model(dist_matrix, r, c0, nugget)
    A[n, n] = 0 # El último elemento de la diagonal es 0
    
    # Construir el vector b (vector de covarianza entre los vecinos y el punto a estimar)
    b_vec = np.ones(n + 1)
    dist_vector = cdist([target_point], coords)[0]
    b_vec[:n] = (c0 + nugget) - spherical_model(dist_vector, r, c0, nugget)
    
    # Resolver el sistema A * w = b para encontrar los pesos w.
    try:
        # Usamos `lstsq` (mínimos cuadrados) ya que es numéricamente más estable que `solve`.
        weights, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        return weights[:n] # Devolvemos solo los pesos de los puntos, no el multiplicador de Lagrange.
    except np.linalg.LinAlgError:
        # En caso de que la matriz sea singular (muy raro), devolver NaNs.
        return np.full(n, np.nan)

# --- Función Principal de Animación ---
variogram_params = V.describe() # Obtener los parámetros una sola vez

def animate(i):
    """Función que se llama para dibujar cada frame de la animación."""
    ax_anim.clear() # Limpiar el frame anterior para un dibujo limpio.
    
    # Re-configurar los ejes y el fondo en cada frame.
    ax_anim.set_xlim(0, 100); ax_anim.set_ylim(0, 100)
    ax_anim.set_title(f'Proceso de Kriging - Estimando Punto {i+1}/{len(points_to_animate)}', fontsize=16)
    ax_anim.set_xlabel('Coordenada X'); ax_anim.set_ylabel('Coordenada Y')
    ax_anim.imshow(kriging_field, origin='lower', extent=(0, 100, 0, 100), cmap='viridis', alpha=0.3)
    ax_anim.scatter(df['x'], df['y'], c='black', s=50, ec='white', label='Muestras')

    # Elementos dinámicos de la animación
    point = points_to_animate[i]
    ax_anim.scatter(point[0], point[1], c='red', s=200, marker='*', ec='white', zorder=10, label='Punto a Estimar')

    # 1. Encontrar vecinos manualmente
    distances = cdist([point], df[['x', 'y']].values)[0]
    neighbor_indices = np.argsort(distances)[:N_NEIGHBORS]
    neighbor_points = df.iloc[neighbor_indices]
    ax_anim.scatter(neighbor_points['x'], neighbor_points['y'], c='cyan', s=150, ec='black', marker='s', zorder=9, label='Vecinos Seleccionados')
    
    # 2. Resolver para obtener los pesos
    weights = solve_kriging_system(point, neighbor_points, variogram_params)
    
    if not np.any(np.isnan(weights)):
        # 3. Calcular la estimación y dibujar los pesos
        estimated_value = np.dot(weights, neighbor_points['grade'])
        for idx, (index, row) in enumerate(neighbor_points.iterrows()):
            ax_anim.text(row['x']+2, row['y']+2, f'w={weights[idx]:.2f}', fontsize=10, weight='bold', color='white',
                         bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        # Dibujar el resultado de la estimación en el punto
        ax_anim.scatter(point[0], point[1], c=estimated_value, s=grid_res*15, marker='s', 
                        cmap='viridis', vmin=vmin, vmax=vmax)
    
    ax_anim.legend(loc='upper right')

# Crear y guardar la animación
ani = animation.FuncAnimation(fig_anim, animate, frames=len(points_to_animate), interval=500, repeat=False)
ani.save('output/kriging_process.gif', writer='pillow', fps=2)
plt.close(fig_anim)

print("\n¡Éxito! Visualizaciones de Kriging generadas.")