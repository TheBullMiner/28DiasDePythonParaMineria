# ==============================================================================
# #28DiasDePythonParaMineria - Día 15
# Título: Validador de Diseño de Rampa con Simulación de Vehículo
# Autor: Maycol Benavides
# The Bull Miner

# Descripción:
# Este script analiza el diseño geométrico de una rampa minera y simula el
# rendimiento de un camión sobre ella. Calcula la velocidad máxima que el
# vehículo puede alcanzar en cada segmento, considerando la pendiente, el peso
# del camión y su curva de potencia (rimpull). El objetivo es identificar
# cuellos de botella en el diseño antes de su construcción.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- 1. PARÁMETROS Y CARGA DE DATOS ---
# En esta sección definimos las constantes del problema: las especificaciones
# del vehículo y las condiciones del camino. También cargamos el diseño de la rampa.
print("[1/4] Cargando datos y configurando la simulación del vehículo...")

# Cargar el diseño de la rampa desde un CSV.
# Se espera un archivo con columnas 'x', 'y', 'z' que definen la línea central.
df = pd.read_csv('data/ramp_design.csv')

# Especificaciones del Camión (Ejemplo: CAT 797F). Estos valores deben ser
# ajustados para el equipo que se desea simular.
EMPTY_WEIGHT = 260 * 1000  # Peso del camión vacío en kg
PAYLOAD = 363 * 1000       # Capacidad de carga máxima en kg
GROSS_WEIGHT_LOADED = EMPTY_WEIGHT + PAYLOAD  # Peso bruto cargado
GROSS_WEIGHT_EMPTY = EMPTY_WEIGHT             # Peso bruto vacío

# Curva de Rendimiento (Rimpull vs. Velocidad). ¡ESTO ES UNA SIMPLIFICACIÓN!
# En un caso real, esta tabla se extrae directamente de las hojas de
# especificaciones del fabricante del camión (Caterpillar, Komatsu, etc.).
# Formato: [velocidad_kph, fuerza_de_tracción_kN]
RIMPULL_CURVE = np.array([
    [10, 1200], [15, 800], [20, 600], [30, 400], [40, 250], [50, 150], [60, 50]
])

# Resistencia a la rodadura (RR) en %. Representa la fricción entre los
# neumáticos y la superficie del camino. 2% es típico para un camino de tierra
# bien mantenido y compactado.
ROLLING_RESISTANCE = 2.0  # %

# --- 2. CÁLCULOS GEOMÉTRICOS ---
# Transformamos la lista de puntos 3D en métricas útiles para el análisis,
# como la distancia acumulada y la pendiente de cada segmento.
print("[2/4] Calculando la geometría de la rampa (distancia y pendiente)...")

# Calcular el cambio en cada coordenada entre puntos consecutivos.
df['dx'] = df['x'].diff()
df['dy'] = df['y'].diff()
df['dz'] = df['z'].diff()

# Calcular la distancia horizontal de cada segmento usando el teorema de Pitágoras.
df['segment_dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
# Calcular la distancia acumulada a lo largo de la rampa. Esto será nuestro eje X.
df['distance_along_ramp'] = df['segment_dist'].cumsum().fillna(0)

# Calcular la pendiente (%) en cada segmento.
# Pendiente = (cambio vertical / cambio horizontal) * 100
df['gradient'] = (df['dz'] / df['segment_dist']) * 100
# Rellenar los valores NaN (que aparecen en la primera fila por el .diff()) con 0.
df.fillna(0, inplace=True)

# --- 3. SIMULACIÓN DE VELOCIDAD DEL VEHÍCULO ---
# Este es el núcleo del análisis. Aquí aplicamos la física básica para
# determinar cómo la geometría de la rampa y el peso del camión afectan su velocidad.
print("[3/4] Simulando la velocidad máxima del camión...")

def calculate_max_speed(weight_kg, gradient_pct):
    """
    Calcula la velocidad máxima teórica de un camión subiendo una pendiente,
    basado en su peso y la curva de rimpull.
    """
    g = 9.81  # Aceleración de la gravedad (m/s^2)
    
    # La fuerza total que el motor debe vencer se compone de dos partes:
    # 1. Resistencia por pendiente (la fuerza de la gravedad que tira del camión hacia atrás)
    grade_resistance_force = weight_kg * g * (gradient_pct / 100) / 1000  # en Kilonewtons (kN)
    
    # 2. Resistencia a la rodadura (la fricción del camino)
    rolling_resistance_force = weight_kg * g * (ROLLING_RESISTANCE / 100) / 1000 # en kN
    
    # La resistencia total es la suma de ambas.
    total_resistance = grade_resistance_force + rolling_resistance_force
    
    # Ahora, usamos la curva de rendimiento para encontrar la velocidad.
    # El camión puede mantener una velocidad si su fuerza de tracción (rimpull)
    # es igual o mayor que la resistencia total.
    speeds = RIMPULL_CURVE[:, 0]
    rimpulls = RIMPULL_CURVE[:, 1]
    
    # Casos extremos:
    if total_resistance > np.max(rimpulls): return 0 # Si la resistencia es demasiado alta, no se puede mover.
    if total_resistance < np.min(rimpulls): return np.max(speeds) # Si la resistencia es muy baja, va a máxima velocidad.
        
    # Usamos interpolación lineal para encontrar la velocidad correspondiente
    # a la resistencia calculada. `np.interp` necesita que el eje x (rimpulls)
    # esté en orden ascendente, por eso usamos `[::-1]`.
    max_speed = np.interp(total_resistance, rimpulls[::-1], speeds[::-1])
    return max_speed

# Aplicamos la función a cada segmento de la rampa para el camión cargado (subiendo).
df['max_speed_loaded_kph'] = df['gradient'].apply(lambda grad: calculate_max_speed(GROSS_WEIGHT_LOADED, grad))

# Simulación de bajada: aquí es una simplificación. En un caso real, se usaría la
# curva de retardo (frenos) del camión. Por ahora, asumimos una velocidad límite por seguridad.
df['max_speed_empty_kph'] = np.where(df['gradient'] < -2, 40, 60) # Más lento en bajadas pronunciadas.

# --- 4. VISUALIZACIÓN MULTICAPA ---
# Construimos el gráfico final capa por capa para una visualización rica en información.
print("[4/4] Creando el perfil de rendimiento de la rampa...")

fig, ax = plt.subplots(figsize=(18, 9))
plt.style.use('seaborn-v0_8-whitegrid')

# Capa 1: El perfil topográfico de la rampa. Es la línea base de nuestro gráfico.
ax.plot(df['distance_along_ramp'], df['z'], color='black', linewidth=3, label='Perfil de la Rampa')

# Capa 2: Un "heatmap" de la pendiente bajo el perfil.
# Creamos una colección de segmentos de línea, donde cada segmento es una parte de
# la rampa, y le asignamos un color basado en su pendiente.
points = np.array([df['distance_along_ramp'], df['z']]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Definimos los colores y los rangos de pendiente para cada color.
cmap = ListedColormap(['#10B981', '#FBBF24', '#EF4444']) # Verde, Amarillo, Rojo
norm = BoundaryNorm([-15, 5, 10, 15], cmap.N) # <5% es verde, 5-10% es amarillo, >10% es rojo

lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(df['gradient'])
lc.set_linewidth(15) # Hacemos la línea muy gruesa para que parezca un fondo de color.
lc.set_alpha(0.7)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax, label='Pendiente (%)')


# Capa 3: Las curvas de rendimiento del vehículo.
ax.plot(df['distance_along_ramp'], df['max_speed_loaded_kph'], 'o--', color='blue', 
        markersize=5, label='Velocidad Máx. Subiendo (Cargado)')
ax.plot(df['distance_along_ramp'], df['max_speed_empty_kph'], 'o--', color='purple', 
        markersize=5, label='Velocidad Máx. Bajando (Vacío)')

# Estilo y etiquetas del gráfico principal.
ax.set_title('Análisis de Rendimiento de Diseño de Rampa', fontsize=20, weight='bold')
ax.set_xlabel('Distancia a lo largo de la Rampa (m)', fontsize=14)
ax.set_ylabel('Elevación (m)', fontsize=14)
ax.legend(loc='upper left')
ax.grid(True)

# Añadir un segundo eje Y en la derecha para mostrar la escala de velocidad.
# `twinx()` crea un eje Y que comparte el mismo eje X.
ax2 = ax.twinx()
ax2.set_ylabel('Velocidad (km/h)', fontsize=14, color='navy')
ax2.tick_params(axis='y', labelcolor='navy')
ax2.set_ylim(0, max(RIMPULL_CURVE[:, 0]) + 10) # El límite se basa en la velocidad máxima del camión.

# Ajustar el layout y guardar la figura.
plt.tight_layout()
plt.savefig('output/ramp_performance_profile.png', dpi=300)

print("\n¡Éxito! Gráfico de análisis de rampa guardado como 'output/ramp_performance_profile.png'.")
