#Maycol Benavides
#The Bull Miner
#Día 08 de Python para Minería

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
# -------------------------------

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
print("[1/5] Cargando datos...")
# Parámetro clave: Radio de influencia de cada pozo en metros
BLAST_RADIUS = 7.5

# Cargar modelo de bloques y diseño de voladura
bm = pd.read_csv('data/block_model.csv')
blast_design = pd.read_csv('data/blast_design.csv')

# --- 2. LÓGICA DE SELECCIÓN DE BLOQUES ("FOOTPRINT") ---
print("[2/5] Identificando bloques dentro de la huella de la voladura...")

# Coordenadas de los centros de los bloques y de los pozos
block_centers_xy = bm[['x', 'y']].values
blast_holes_xy = blast_design[['x', 'y']].values

# Calcular la distancia horizontal de cada bloque al pozo más cercano
min_dist_to_hole = cdist(block_centers_xy, blast_holes_xy).min(axis=1)

# Determinar el rango vertical de la voladura
z_min_blast = blast_design['z_collar'].min() - blast_design['depth'].max()
z_max_blast = blast_design['z_collar'].max()

# Crear una máscara booleana para seleccionar los bloques
is_in_footprint = (min_dist_to_hole <= BLAST_RADIUS) & \
                  (bm['z'] >= z_min_blast) & \
                  (bm['z'] <= z_max_blast)

selected_blocks = bm[is_in_footprint].copy()

# --- 3. CÁLCULO DE KPIs ---
print("[3/5] Calculando KPIs para el material seleccionado...")

selected_blocks['block_volume'] = selected_blocks['dx'] * selected_blocks['dy'] * selected_blocks['dz']
selected_blocks['block_tons'] = selected_blocks['block_volume'] * selected_blocks['density']
selected_blocks['metal_tons'] = selected_blocks['block_tons'] * (selected_blocks['cu_grade'] / 100)

total_tons = selected_blocks['block_tons'].sum()
total_metal = selected_blocks['metal_tons'].sum()
avg_grade = (total_metal / total_tons) * 100 if total_tons > 0 else 0
waste_tons = selected_blocks[selected_blocks['rock_type'] == 'Waste']['block_tons'].sum()
dilution_pct = (waste_tons / total_tons) * 100 if total_tons > 0 else 0

# --- 4. VISUALIZACIÓN ---
print("[4/5] Generando los gráficos...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# --- Panel 1: Vista en Planta ---
Z_SLICE = 95
plan_view_bm = bm[bm['z'] == Z_SLICE]

sc = ax1.scatter(plan_view_bm['x'], plan_view_bm['y'], c=plan_view_bm['cu_grade'],
                 cmap='viridis', s=250, marker='s', ec='gray', lw=0.5)
plt.colorbar(sc, ax=ax1, label='Ley de Cobre (%)')

hull = ConvexHull(blast_holes_xy)
for simplex in hull.simplices:
    ax1.plot(blast_holes_xy[simplex, 0], blast_holes_xy[simplex, 1], 'r-', lw=2)

ax1.scatter(blast_design['x'], blast_design['y'], c='red', s=50, ec='black', label='Pozos de Voladura')
ax1.set_title(f'Vista en Planta (Elevación {Z_SLICE}m)', fontsize=16, weight='bold')
ax1.set_xlabel('Coordenada Este (m)')
ax1.set_ylabel('Coordenada Norte (m)')
ax1.set_aspect('equal', adjustable='box')
ax1.legend()

# --- Panel 2: Vista en Sección ---
X_SLICE = 50
section_view_bm = bm[bm['x'] == X_SLICE]
colors = {'Ore': 'orange', 'Waste': 'gray'}

ax2.scatter(section_view_bm['y'], section_view_bm['z'], c=section_view_bm['rock_type'].map(colors),
            s=250, marker='s', ec='black', lw=0.5, alpha=0.7)

for _, hole in blast_design[blast_design['x'] == X_SLICE].iterrows():
    rect = Rectangle((hole['y'] - BLAST_RADIUS, hole['z_collar'] - hole['depth']),
                     2 * BLAST_RADIUS, hole['depth'],
                     facecolor='red', alpha=0.3, ec='red', lw=1.5, linestyle='--')
    ax2.add_patch(rect)

ax2.set_title(f'Vista en Sección (Corte en X={X_SLICE}m)', fontsize=16, weight='bold')
ax2.set_xlabel('Coordenada Norte (m)')
ax2.set_ylabel('Elevación (m)')
ax2.set_aspect('equal', adjustable='box')

legend_patches = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
ax2.legend(legend_patches, colors.keys(), title='Tipo de Roca')

# --- Anotación con KPIs ---
kpi_text = f"""
Resultados de la Voladura:
--------------------------
Toneladas Totales: {total_tons:,.0f} t
Ley Ponderada Cu: {avg_grade:.2f}%
Toneladas de Cobre Fino: {total_metal:,.0f} t
Dilución: {dilution_pct:.1f}%
"""
fig.text(0.5, 0.05, kpi_text, ha='center', va='bottom', fontsize=14,
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# --- 5. GUARDAR ---
print("[5/5] Guardando el gráfico...")
plt.savefig('output/blast_footprint_analysis.png', dpi=300)
print("\n¡Éxito! Análisis de huella de voladura completado.")