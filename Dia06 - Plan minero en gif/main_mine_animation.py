#Maycol Benavides
#The Bull Miner
#28Días de Python para Minería

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import os

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
print("[1/4] Cargando el plan minero...")
DATA_PATH = 'data'
OUTPUT_PATH = 'output'
os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_PATH, 'mine_plan.csv'))

# --- 2. PRE-PROCESAMIENTO DE DATOS (VERSIÓN CORREGIDA Y ROBUSTA) ---
print("[2/4] Pre-procesando datos para la animación...")

# Agrupar las toneladas por año y ubicación
tons_per_year = df.groupby(['year', 'location_id'])['tons'].sum().reset_index()

### CAMBIO CLAVE: Crear un DataFrame con el estado acumulado para cada año ###

# Obtener todas las combinaciones únicas de años y ubicaciones
all_years = sorted(df['year'].unique())
all_locations = df['location_id'].unique()
idx = pd.MultiIndex.from_product([all_years, all_locations], names=['year', 'location_id'])

# Reindexar nuestros datos para tener una fila para cada ubicación en cada año (rellenando con 0 si no hay movimiento)
plan_full = tons_per_year.set_index(['year', 'location_id']).reindex(idx, fill_value=0).reset_index()

# Calcular el acumulado
plan_full['cumulative_tons'] = plan_full.groupby('location_id')['tons'].cumsum()

# Añadir las propiedades fijas (coordenadas y tipo de material)
locations_props = df[['location_id', 'x', 'y', 'material_type']].drop_duplicates()
plan_cumulative_df = pd.merge(plan_full, locations_props, on='location_id')


# Escala para que el tamaño de los círculos sea visible. ¡AJUSTA ESTE VALOR!
SCALE_FACTOR = 0.1 

# --- 3. CONFIGURACIÓN DEL GRÁFICO Y LA ANIMACIÓN ---
print("[3/4] Configurando el lienzo de la animación...")

fig, ax = plt.subplots(figsize=(10, 8))

# Función que se llama para cada frame (cada año) de la animación
def update(year):
    ax.clear()  # Limpiar el frame anterior

    # Filtrar el DataFrame pre-procesado para el año actual
    data_for_year = plan_cumulative_df[plan_cumulative_df['year'] == year]
    
    total_tons_moved = data_for_year['cumulative_tons'].sum()

    # Dibujar cada ubicación con su tamaño acumulado para ese año
    for _, row in data_for_year.iterrows():
        tons = row['cumulative_tons']
        if tons > 0:
            radius = np.sqrt(tons / np.pi) * SCALE_FACTOR
            color = 'saddlebrown' if row['material_type'] == 'Waste' else 'darkslategrey'
            
            circle = plt.Circle((row['x'], row['y']), radius, color=color, alpha=0.8, ec='black')
            ax.add_patch(circle)
            ax.text(row['x'], row['y'], row['location_id'], ha='center', va='center', color='white', weight='bold', fontsize=8)

    # Re-establecer límites y estilo en cada frame
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Evolución del Plan Minero - Año: {year}\nTotal Acumulado: {total_tons_moved:,.0f} T", fontsize=16)
    ax.set_xlabel("Coordenada Este")
    ax.set_ylabel("Coordenada Norte")
    ax.grid(True, linestyle='--', alpha=0.5)

# --- 4. CREACIÓN Y GUARDADO DEL GIF ---
# Crear el objeto de animación
ani = animation.FuncAnimation(fig, update, frames=all_years, repeat=True)

# Guardar el GIF
output_filename = os.path.join(OUTPUT_PATH, 'mine_plan_evolution_fixed.gif')
print(f"[4/4] Creando y guardando el GIF en '{output_filename}'. Esto puede tardar unos segundos...")
ani.save(output_filename, writer='pillow', fps=1.5)

print("\n¡Éxito! GIF animado del plan minero generado.")