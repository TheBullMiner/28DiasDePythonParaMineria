import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

# --- 1. CARGA Y PREPARACIÓN DE DATOS ---
print("[1/4] Cargando y preparando datos del sondaje...")

# Cargar los datos
df = pd.read_csv('data/drillhole_data.csv')

# Para el análisis, necesitamos un array de NumPy.
# Vamos a analizar la variación conjunta de Cobre y Molibdeno.
# Normalizamos los datos para que ambas variables tengan una escala similar.
signals = df[['cu_grade', 'mo_ppm']].values
signals_normalized = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)

# --- 2. DETECCIÓN DE PUNTOS DE CAMBIO ---
print("[2/4] Aplicando algoritmo de detección de cambios...")

# Modelo a usar: Detección Binaria por Segmentación (Binary Segmentation)
# Costo: L2 (distancia euclidiana), ya que buscamos cambios en la media.
# Número de puntos de cambio a detectar:
# Podríamos definirlo nosotros (ej. n_bkps=2) o dejar que el algoritmo lo estime.
# Usaremos un "penalty" para que el algoritmo decida.
algo = rpt.Binseg(model="l2").fit(signals_normalized)
change_points_indices = algo.predict(pen=np.log(len(signals_normalized)) * 1.5) # El valor de pen es clave

# Los índices que devuelve ruptures apuntan al final del segmento.
# Convertimos los índices de fila a profundidades 'to'.
change_depths = [df['to'].iloc[i-1] for i in change_points_indices if i < len(df)]

print(f"Puntos de cambio detectados en las profundidades (aprox): {change_depths}")

# --- 3. PREPARACIÓN PARA LA VISUALIZACIÓN ---
print("[3/4] Preparando los gráficos del log de sondaje...")

# Mapeo de tipo de roca a un color y un número para el gráfico
unique_rocks = df['rock_type'].unique()
rock_map = {rock: i for i, rock in enumerate(unique_rocks)}
color_map = plt.cm.get_cmap('Dark2', len(unique_rocks))

df['rock_code'] = df['rock_type'].map(rock_map)

# --- 4. CREACIÓN DEL GRÁFICO (LOG DE SONDALE) ---
print("[4/4] Generando el log de sondaje con los contactos detectados...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 3, figsize=(10, 15), sharey=True,
                        gridspec_kw={'width_ratios': [1, 2, 2]})

fig.suptitle('Análisis de Contactos Geológicos con Detección de Cambios', fontsize=16, weight='bold')

# --- Panel 1: Litología ---
ax1 = axs[0]
for _, row in df.iterrows():
    ax1.axhspan(row['from'], row['to'], facecolor=color_map(row['rock_code']), alpha=0.8)
ax1.set_ylabel('Profundidad (m)', fontsize=12)
ax1.set_title('Litología', fontsize=12)
# Crear una leyenda para la litología
legend_patches = [plt.Rectangle((0,0),1,1, color=color_map(rock_map[rock])) for rock in unique_rocks]
ax1.legend(legend_patches, unique_rocks, loc='upper left')

# --- Panel 2: Ley de Cobre ---
ax2 = axs[1]
ax2.plot(df['cu_grade'], df['to'], 'b-', label='Cu (%)')
ax2.set_xlabel('Ley de Cobre (%)')
ax2.set_title('Cobre', fontsize=12)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Panel 3: Ley de Molibdeno ---
ax3 = axs[2]
ax3.plot(df['mo_ppm'], df['to'], 'g-', label='Mo (ppm)')
ax3.set_xlabel('Molibdeno (ppm)')
ax3.set_title('Molibdeno', fontsize=12)
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Añadir los contactos detectados en todos los paneles ---
for depth in change_depths:
    for ax in axs:
        ax.axhline(y=depth, color='r', linestyle='--', linewidth=2.5, label=f'Contacto Detectado @ {depth}m')

# Añadir una leyenda única para la línea de contacto
handles, labels = axs[-1].get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

# Invertir el eje Y para que la profundidad aumente hacia abajo
ax1.invert_yaxis()

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Ajustar para el título y la leyenda

# Guardar el gráfico
output_filename = 'output/drillhole_log_analysis.png'
plt.savefig(output_filename, dpi=300)

print(f"\n¡Éxito! Gráfico del log de sondaje guardado en '{output_filename}'.")