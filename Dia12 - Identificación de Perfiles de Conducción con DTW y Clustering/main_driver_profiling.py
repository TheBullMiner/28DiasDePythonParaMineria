# ==============================================================================
# #28DiasDePythonParaMineria - Día 12
# Título: Identificación de Perfiles de Conducción con DTW y Clustering
# Autor: Maycol Benavides 
# The Bull Miner
#
# Descripción:
# Este script utiliza técnicas de Machine Learning para series de tiempo para
# analizar y clasificar los perfiles de velocidad de los viajes de una flota de
# camiones. El objetivo es identificar automáticamente "estilos de conducción"
# (ej. agresivo, eficiente, cauteloso) sin supervisión previa.
#
# Técnica Clave:
# - Dynamic Time Warping (DTW): Una métrica para medir la similitud entre dos
#   series de tiempo que pueden variar en velocidad o duración.
# - TimeSeriesKMeans: Un algoritmo de clustering K-Means que utiliza DTW como
#   su función de distancia para agrupar series de tiempo similares.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

# --- 1. CARGA Y PREPARACIÓN DE DATOS ---
# El primer paso es leer los datos crudos y transformarlos a un formato
# que la librería `tslearn` pueda entender.
print("[1/4] Cargando y preparando los datos de los viajes...")

# Cargar los datos desde un archivo CSV.
# Se espera que el CSV tenga columnas 'trip_id', 'time_step', y 'speed'.
df = pd.read_csv('data/truck_trips.csv')

# Agrupar el DataFrame por 'trip_id' para separar cada viaje.
# Convertimos la columna 'speed' de cada viaje en un array de NumPy
# y lo guardamos en una lista. El resultado es una lista de arrays, donde
# cada array es una serie de tiempo de velocidad de un viaje.
trips = []
for trip_id, trip_data in df.groupby('trip_id'):
    trips.append(trip_data['speed'].values)

# `tslearn` requiere que todas las series de tiempo en un dataset tengan la misma longitud.
# La función `to_time_series_dataset` se encarga de esto: toma nuestra lista de
# viajes (que tienen diferentes longitudes) y crea un único array 3D,
# rellenando las series más cortas con `NaN` (Not a Number) al final.
formatted_dataset = to_time_series_dataset(trips)

# --- 2. CLUSTERING CON DTW ---
# Aquí es donde ocurre la magia del Machine Learning.
# Usaremos un algoritmo de clustering para que agrupe los viajes por sí mismo.
print("[2/4] Ejecutando clustering con Dynamic Time Warping (DTW)...")

# Definimos cuántos perfiles de conducción o "clusters" queremos que el algoritmo encuentre.
N_CLUSTERS = 3

# Inicializamos el modelo de K-Means para Series de Tiempo.
# - n_clusters: El número de grupos a encontrar (definido arriba).
# - metric="dtw": ¡Este es el paso crucial! Le decimos al algoritmo que no use
#   la distancia Euclidiana estándar, sino Dynamic Time Warping. Esto le permite
#   comparar formas de curvas de manera flexible.
# - random_state: Para que los resultados sean reproducibles.
# - n_init: El algoritmo se ejecuta varias veces con diferentes inicios para
#   encontrar una solución más estable y robusta.
dtw_km = TimeSeriesKMeans(n_clusters=N_CLUSTERS,
                         metric="dtw",
                         verbose=False, 
                         random_state=42,
                         n_init=3)

# Entrenamos el modelo con nuestros datos formateados y predecimos la etiqueta
# de cluster para cada viaje. `cluster_labels` será un array como [0, 1, 1, 2, 0, ...].
cluster_labels = dtw_km.fit_predict(formatted_dataset)

# --- 3. ANÁLISIS DE RESULTADOS ---
# El modelo nos ha dado los grupos. Ahora, les damos un nombre y los analizamos.
print("[3/4] Analizando los perfiles de los clusters...")

# Creamos un diccionario para mapear el número del cluster (0, 1, 2) a un nombre
# descriptivo. Estos nombres se asignan después de observar los gráficos y entender
# qué representa cada cluster.
PROFILE_NAMES = {0: "Perfil Agresivo", 1: "Perfil Eficiente/Estándar", 2: "Perfil Cauteloso/Interrumpido"}

# Usamos pandas para contar fácilmente cuántos viajes cayeron en cada cluster.
cluster_counts = pd.Series(cluster_labels).value_counts()
print("\n--- Conteo de Viajes por Perfil ---")
for i in sorted(cluster_counts.index): # Usamos sorted para un orden de impresión consistente.
    print(f"{PROFILE_NAMES.get(i, f'Cluster {i}')}: {cluster_counts[i]} viajes")

# --- 4. VISUALIZACIÓN DE LOS PERFILES ---
# La mejor forma de entender los resultados del clustering es visualizándolos.
print("[4/4] Creando el gráfico de perfiles de conducción...")

# Creamos una figura con N_CLUSTERS subplots, uno para cada perfil.
# `sharex` y `sharey` hacen que todos los subplots tengan los mismos ejes, facilitando la comparación.
fig, axs = plt.subplots(N_CLUSTERS, 1, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle('Identificación de Perfiles de Conducción con DTW', fontsize=22, weight='bold')

# Generamos un mapa de colores para que cada cluster tenga un color distintivo.
colors = cm.viridis(np.linspace(0, 1, N_CLUSTERS))

# Iteramos sobre cada cluster para dibujarlo en su propio subplot.
for i in range(N_CLUSTERS):
    ax = axs[i]
    
    # Dibujar cada viaje individual que pertenece a este cluster.
    # Usamos una opacidad baja (`alpha=0.2`) para que las líneas individuales
    # se mezclen y nos permitan ver la densidad y variabilidad del cluster.
    for j, trip in enumerate(trips):
        if cluster_labels[j] == i:
            ax.plot(trip, color=colors[i], alpha=0.2)
            
    # El modelo `dtw_km` calcula un "centroide" para cada cluster, que es
    # la serie de tiempo promedio o más representativa del grupo.
    # La dibujamos como una línea negra gruesa para que destaque.
    cluster_center = dtw_km.cluster_centers_[i].ravel()
    ax.plot(cluster_center, color='black', linewidth=3)

    # Añadir títulos y etiquetas a cada subplot.
    ax.set_title(PROFILE_NAMES.get(i, f'Cluster {i}'), fontsize=16)
    ax.set_ylabel('Velocidad (km/h)')
    ax.grid(True, linestyle='--', alpha=0.6)

# Añadir la etiqueta del eje X solo al último subplot.
axs[-1].set_xlabel('Paso de Tiempo (secuencia de datos)')

# Ajustar el layout y guardar la figura final en alta resolución.
plt.tight_layout(rect=[0, 0, 1, 0.96]) # `rect` deja espacio para el supertítulo.
plt.savefig('output/driver_profiles.png', dpi=300)

print("\n¡Éxito! Gráfico de perfiles guardado como 'output/driver_profiles.png'.")