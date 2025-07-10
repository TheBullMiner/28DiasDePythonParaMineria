# ==============================================================================
# #28DiasDePythonParaMineria - Día 23
# Título: Animación de un Gráfico de Control Estadístico de Procesos (SPC)
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script simula un dashboard de monitoreo de procesos en tiempo real.
# Lee una serie de tiempo de un proceso (ej. ley de concentrado), calcula
# los límites de control estadístico basados en un periodo estable, y luego
# genera una animación que muestra los datos apareciendo punto por punto.
# Cuando un punto viola una regla de control (sale de los límites), el
# script genera una alerta visual.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
# El primer paso es cargar los datos del proceso y calcular los parámetros
# estadísticos que definirán nuestro gráfico de control.
print("[1/4] Cargando datos y calculando límites de control...")
df = pd.read_csv('data/process_data.csv')

# Para calcular los límites de control, es crucial usar un periodo donde
# se sabe que el proceso estaba operando de manera estable y "normal".
# Aquí, usamos los primeros 30 puntos como nuestro periodo de calibración.
stable_data = df['concentrate_grade'].iloc[:30]
mean = stable_data.mean()
std_dev = stable_data.std()

# Definimos los límites de control basados en las reglas de Shewhart.
# Estos límites nos ayudan a distinguir la variación común de la variación especial.
UCL = mean + 3 * std_dev  # Límite de Control Superior (Upper Control Limit)
LCL = mean - 3 * std_dev  # Límite de Control Inferior (Lower Control Limit)
UWL = mean + 2 * std_dev  # Límite de Advertencia Superior (Upper Warning Limit)
LWL = mean - 2 * std_dev  # Límite de Advertencia Inferior (Lower Warning Limit)

# --- Crear la carpeta de salida si no existe ---
# Una buena práctica para evitar errores de 'FileNotFoundError'.
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta '{OUTPUT_DIR}' creada.")

# --- 2. VISUALIZACIÓN ESTÁTICA FINAL ---
# Creamos una imagen del gráfico completo al final del periodo. Sirve como
# resumen y es la imagen estática perfecta para el post.
print("[2/4] Creando la imagen estática del gráfico de control final...")
fig_static, ax_static = plt.subplots(figsize=(15, 8))

# Dibujar las series de datos y las líneas de control.
ax_static.plot(df['hour'], df['concentrate_grade'], 'o-', color='black', label='Ley de Concentrado')
ax_static.axhline(mean, color='blue', linestyle='-', label='Media del Proceso')
ax_static.axhline(UCL, color='red', linestyle='--', label='LCS (+3σ)')
ax_static.axhline(LCL, color='red', linestyle='--', label='LCI (-3σ)')
ax_static.axhline(UWL, color='orange', linestyle=':', label='LAS (+2σ)')
ax_static.axhline(LWL, color='orange', linestyle=':', label='LAI (-2σ)')

# Identificar y resaltar visualmente los puntos que están fuera de control.
out_of_control = df[(df['concentrate_grade'] > UCL) | (df['concentrate_grade'] < LCL)]
ax_static.scatter(out_of_control['hour'], out_of_control['concentrate_grade'],
                  s=150, facecolors='none', edgecolors='red', linewidth=2, label='Fuera de Control')

# Añadir títulos y etiquetas para claridad.
ax_static.set_title('Gráfico de Control Estadístico de Procesos (SPC)', fontsize=16)
ax_static.set_xlabel('Hora')
ax_static.set_ylabel('Ley de Concentrado de Cu (%)')
ax_static.legend()
ax_static.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(OUTPUT_DIR, 'spc_chart_final.png'), dpi=150)
plt.close(fig_static)


# --- 3. CREACIÓN DEL GIF ANIMADO ---
# Aquí es donde ocurre la magia de la simulación "en tiempo real".
print("[3/4] Configurando la animación del monitoreo en tiempo real...")
fig_anim, ax_anim = plt.subplots(figsize=(15, 8))

def animate(i):
    """
    Función de animación que se llama para cada frame (cada punto de dato).
    'i' es el número del frame, que usamos como índice para los datos.
    """
    ax_anim.clear() # Limpiar el gráfico anterior para dibujar el nuevo estado.
    
    # Seleccionar los datos desde el inicio hasta el frame actual.
    current_data = df.iloc[:i+1]
    
    # Dibujar las líneas de control estáticas en cada frame.
    ax_anim.axhline(mean, color='blue', linestyle='-')
    ax_anim.axhline(UCL, color='red', linestyle='--')
    ax_anim.axhline(LCL, color='red', linestyle='--')
    
    # Dibujar la línea de proceso hasta el momento actual.
    ax_anim.plot(current_data['hour'], current_data['concentrate_grade'], 'o-', color='black', markersize=5)
    
    # --- Lógica de Alerta Visual ---
    # Obtener el último punto de dato añadido.
    last_point = current_data.iloc[-1]
    # Comprobar si este último punto viola la regla de control más básica.
    is_out_of_control = (last_point['concentrate_grade'] > UCL) or (last_point['concentrate_grade'] < LCL)
    
    if is_out_of_control:
        # Si está fuera de control, lo resaltamos.
        # 1. Dibujar un círculo rojo grande y brillante sobre el punto.
        ax_anim.scatter(last_point['hour'], last_point['concentrate_grade'], s=300, c='red', ec='black', zorder=10)
        
        # 2. Añadir un texto de alerta que parpadea.
        # El parpadeo se logra cambiando la visibilidad del texto en frames pares e impares.
        alert_visibility = (i % 2 == 0)
        ax_anim.text(0.5, 0.9, '¡ALERTA: PROCESO FUERA DE CONTROL!', ha='center', va='center',
                     transform=ax_anim.transAxes, fontsize=18, color='red', weight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, ec='red'), visible=alert_visibility)

    # Configuración de los ejes y títulos para mantener la consistencia en cada frame.
    ax_anim.set_title('Monitoreo de Proceso en Tiempo Real', fontsize=16)
    ax_anim.set_xlabel('Hora')
    ax_anim.set_ylabel('Ley de Concentrado de Cu (%)')
    ax_anim.set_xlim(0, len(df) + 1)
    ax_anim.set_ylim(LCL - 1, UCL + 1) # Asegurar que los límites siempre sean visibles.
    ax_anim.grid(True, linestyle='--', alpha=0.6)

# --- 4. GUARDADO DEL GIF ---
print("[4/4] Creando y guardando el GIF animado...")

# Crear el objeto de animación. `blit=False` es más robusto cuando los
# elementos del gráfico (como textos) cambian de forma compleja.
ani = animation.FuncAnimation(fig_anim, animate, frames=len(df),
                              interval=150, blit=False)
                              
# Guardar la animación como un archivo GIF.
ani.save(os.path.join(OUTPUT_DIR, 'spc_chart_live.gif'), writer='pillow', fps=10)
plt.close(fig_anim)

print("\n¡Éxito! Visualizaciones de SPC generadas.")