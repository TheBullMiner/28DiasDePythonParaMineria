# ==============================================================================
# #28DiasDePythonParaMineria - Día 24
# Título: Análisis de "Shovel Hang Time" con Heatmap de Calendario
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script analiza los datos de estado de una pala para cuantificar el
# "Hang Time", que es el tiempo que la pala pasa inactiva esperando por un
# camión. Agrega este tiempo perdido por día y genera dos visualizaciones:
# 1. Un heatmap de calendario estático para identificar patrones diarios y semanales.
# 2. Un GIF animado que muestra el impacto acumulado del hang time a lo largo
#    de un mes.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import calmap
import os

# --- 1. CARGA Y PROCESAMIENTO DE DATOS ---
# El primer paso es leer los datos de eventos y calcular la duración de cada estado.
print("[1/4] Cargando y procesando los datos de estado de la pala...")
df = pd.read_csv('data/shovel_status.csv', parse_dates=['timestamp'])

# Asegurarse de que los datos están ordenados cronológicamente.
df = df.sort_values('timestamp').reset_index(drop=True)

# Calcular la duración de cada estado encontrando la diferencia de tiempo
# con el siguiente evento. `shift(-1)` mueve la fila siguiente hacia arriba.
df['duration_s'] = df['timestamp'].diff().dt.total_seconds().shift(-1)
df.dropna(inplace=True) # Eliminar la última fila que tendrá NaN en duración.

# Filtrar solo los eventos que nos interesan: la espera por camión.
df_waits = df[df['status'] == 'Waiting for Truck'].copy()
df_waits['hang_time_min'] = df_waits['duration_s'] / 60

# --- 2. AGREGACIÓN DE DATOS DIARIOS ---
# Agrupamos los datos para obtener una única métrica por día.
print("[2/4] Agregando el 'Hang Time' total por día...")

# Agrupar por fecha y sumar todos los minutos de hang time para cada día.
daily_hang_time = df_waits.groupby(df_waits['timestamp'].dt.date)['hang_time_min'].sum()
daily_hang_time.index = pd.to_datetime(daily_hang_time.index) # Convertir el índice a DatetimeIndex

# Calcular la suma acumulada para la animación del GIF.
daily_hang_time_cumulative = daily_hang_time.cumsum()

# Crear la carpeta de salida si no existe.
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta '{OUTPUT_DIR}' creada.")

# --- 3. VISUALIZACIÓN ESTÁTICA (CON BARRA DE COLOR) ---
# Creamos el heatmap de calendario que es perfecto para identificar patrones.
print("[3/4] Creando el heatmap de calendario estático mejorado...")

fig_static, ax_static = plt.subplots(figsize=(20, 8))
calmap.yearplot(daily_hang_time, year=2023, cmap='YlOrRd', linewidth=2, ax=ax_static)
fig_static.suptitle('Análisis de "Shovel Hang Time" (Minutos Totales de Espera por Día)', fontsize=20, weight='bold')

# Añadir una barra de color manualmente para que el gráfico sea cuantitativo.
norm = plt.Normalize(vmin=daily_hang_time.min(), vmax=daily_hang_time.max())
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
sm.set_array([]) # Truco necesario para que funcione la colorbar sin una imagen base.
cbar = fig_static.colorbar(sm, ax=ax_static, orientation='vertical', pad=0.02, shrink=0.7)
cbar.set_label('Minutos de Espera (Hang Time)', rotation=270, labelpad=15)

plt.savefig(os.path.join(OUTPUT_DIR, 'shovel_hang_time_final.png'), dpi=150, bbox_inches='tight')
plt.close(fig_static)

# --- 4. CREACIÓN DEL GIF ANIMADO (GRÁFICO ACUMULADO) ---
# Esta animación cuenta una historia diferente: el impacto acumulado del tiempo perdido.
print("[4/4] Creando el GIF animado del 'Hang Time' acumulado...")

fig_anim, ax_anim = plt.subplots(figsize=(12, 7))

# Configuración inicial del gráfico de línea que se irá "dibujando".
line, = ax_anim.plot([], [], 'o-', color='#DC2626', markersize=5)
ax_anim.set_xlim(daily_hang_time_cumulative.index.min(), daily_hang_time_cumulative.index.max())
ax_anim.set_ylim(0, daily_hang_time_cumulative.max() * 1.1)
ax_anim.set_title("Acumulado de 'Shovel Hang Time' a lo largo del Mes", fontsize=16)
ax_anim.set_xlabel("Fecha")
ax_anim.set_ylabel("Minutos Totales de Espera Acumulados")
ax_anim.grid(True, linestyle='--')
plt.xticks(rotation=45)

# Texto dinámico que mostrará la fecha y el valor acumulado en cada frame.
time_text = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes, fontsize=14,
                         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Función de animación que se llama para cada frame (cada día).
def animate(i):
    """Actualiza el gráfico de línea hasta el día 'i'."""
    data_to_plot = daily_hang_time_cumulative.iloc[:i+1]
    line.set_data(data_to_plot.index, data_to_plot.values)
    
    # Actualizar el texto dinámico.
    date_str = data_to_plot.index[-1].strftime('%Y-%m-%d')
    cumulative_minutes = data_to_plot.values[-1]
    time_text.set_text(f'Fecha: {date_str}\nAcumulado: {cumulative_minutes:.0f} min')
    return line, time_text

# Crear y guardar la animación. `blit=True` es una optimización para un renderizado más rápido.
ani = animation.FuncAnimation(fig_anim, animate, frames=len(daily_hang_time_cumulative),
                              interval=150, blit=True)
ani.save(os.path.join(OUTPUT_DIR, 'shovel_hang_time_cumulative.gif'), writer='pillow', fps=10)
plt.close(fig_anim)

print("\n¡Éxito! Visualizaciones de Hang Time generadas.")