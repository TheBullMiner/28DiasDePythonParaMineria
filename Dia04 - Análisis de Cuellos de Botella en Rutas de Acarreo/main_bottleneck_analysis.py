import pandas as pd
import geopandas
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
print("[1/4] Cargando datos y configurando el entorno...")
DATA_PATH = 'data'
OUTPUT_PATH = 'output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_PATH, 'truck_gps_data.csv'))

# --- 2. CONVERSIÓN A GEOESPACIAL Y CREACIÓN DE SEGMENTOS ---
print("[2/4] Procesando datos GPS a formato geoespacial...")
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))
gdf.set_crs("EPSG:4326", inplace=True)
gdf = gdf.to_crs(epsg=32719) # ¡Recuerda ajustar tu zona UTM!

segments = []
for truck_id, truck_data in gdf.groupby('truck_id'):
    truck_data = truck_data.sort_values('timestamp')
    points = truck_data.geometry.tolist()
    speeds = truck_data['speed_kmh'].tolist()
    for i in range(len(points) - 1):
        segment = LineString([points[i], points[i+1]])
        avg_speed = (speeds[i] + speeds[i+1]) / 2
        segments.append({'geometry': segment, 'avg_speed': avg_speed})

segments_gdf = geopandas.GeoDataFrame(segments, crs="EPSG:32719")

# --- 3. GENERACIÓN DEL GRÁFICO "HOT" (SECCIÓN CORREGIDA) ---
print("[3/4] Creando el mapa de calor de velocidades...")

fig, ax = plt.subplots(1, 1, figsize=(15, 12))

### CAMBIO CLAVE 1: CALCULAR LÍMITES DINÁMICAMENTE ###
# Obtenemos las coordenadas mínimas y máximas de todas nuestras rutas.
# Añadimos un pequeño buffer (margen) para que no quede tan ajustado.
buffer = 200 # en metros
xmin, ymin, xmax, ymax = segments_gdf.total_bounds
xmin_buf, ymin_buf = xmin - buffer, ymin - buffer
xmax_buf, ymax_buf = xmax + buffer, ymax + buffer

# OPCIONAL PERO RECOMENDADO: Añadir imagen de fondo
try:
    mine_layout_img = plt.imread(os.path.join(DATA_PATH, 'mine_layout.png'))
    # Usamos los límites calculados dinámicamente como el nuevo 'extent'
    ax.imshow(mine_layout_img, extent=[xmin_buf, xmax_buf, ymin_buf, ymax_buf], aspect='auto')
except FileNotFoundError:
    print("Advertencia: No se encontró 'mine_layout.png'. El gráfico se generará sobre fondo blanco.")

# Dibujar los segmentos de ruta, coloreados por velocidad
segments_gdf.plot(ax=ax, 
                  column='avg_speed', 
                  cmap='RdYlGn', 
                  linewidth=3, 
                  legend=True,
                  legend_kwds={'label': "Velocidad Promedio (km/h)",
                               'orientation': "vertical",
                               'shrink': 0.8})

### CAMBIO CLAVE 2: ESTABLECER LÍMITES DEL GRÁFICO ###
# Forzamos al gráfico a tener los mismos límites que nuestra imagen
ax.set_xlim(xmin_buf, xmax_buf)
ax.set_ylim(ymin_buf, ymax_buf)

# Estilo del gráfico
ax.set_title('Análisis de Cuellos de Botella en Rutas de Acarreo', fontsize=20, weight='bold')
ax.set_xlabel('Coordenada Este (m)')
ax.set_ylabel('Coordenada Norte (m)')
ax.set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)

# --- 4. GUARDAR EL RESULTADO ---
print("[4/4] Guardando el gráfico en alta resolución...")
output_filename = os.path.join(OUTPUT_PATH, 'speed_heatmap_fixed.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"\n¡Éxito! Mapa de calor corregido generado en: {output_filename}")