# ==============================================================================
# #28DiasDePythonParaMineria - Día 19
# Título: Simulador de Propagación de Contaminantes en Aguas Subterráneas
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script modela el transporte de un contaminante en un acuífero 2D
# utilizando la ecuación de Advección-Dispersión. Se resuelve numéricamente
# mediante un método de diferencias finitas en una grilla. El objetivo es
# visualizar cómo una pluma de contaminante se mueve (advección) y se expande
# (dispersión) a lo largo del tiempo.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

# --- 1. CONFIGURACIÓN DEL MODELO (VERSIÓN "CINEMATOGRÁFICA") ---
# Aquí definimos las constantes físicas y de la simulación. Ajustar estos
# valores permite modelar diferentes tipos de acuíferos y escenarios.
print("[1/5] Configurando los parámetros del modelo...")

# Parámetros de la grilla y del tiempo
NX, NY = 151, 101       # Dimensiones de la grilla (celdas). Más ancha para ver mejor la pluma.
DX, DY = 10, 10         # Tamaño de cada celda (metros).
NT = 1200               # Número total de pasos de tiempo a simular.
DT = 1.0 * 24 * 3600    # Paso de tiempo en segundos (1 día). Un paso de tiempo estable.

# Parámetros hidrogeológicos del acuífero
POROSITY = 0.3          # Porosidad efectiva (adimensional). Afecta la velocidad real del flujo.
D_L = 0.1               # Coeficiente de dispersión longitudinal (m^2/s). Se redujo para una pluma más definida.
D_T = 0.01              # Coeficiente de dispersión transversal (m^2/s). Se redujo para evitar que se ensanche demasiado rápido.
V_X = 2.5e-6            # Velocidad del agua en dirección X (m/s). Reducida para una evolución más lenta.
V_Y = 0.5e-6            # Velocidad del agua en dirección Y (m/s).

# Parámetros de la fuente de contaminación
SOURCE_X, SOURCE_Y = 25, 50 # Posición (índice de celda) de la fuente de la fuga.
SOURCE_STRENGTH = 100.0     # Concentración inicial en la fuente (unidades arbitrarias).
SOURCE_DURATION_STEPS = 150 # La fuga dura los primeros 150 pasos de tiempo (150 días).

# Crear la carpeta de salida si no existe. Es una buena práctica para evitar errores.
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. INICIALIZACIÓN DE LA GRILLA ---
# Creamos un array de NumPy que representará nuestro acuífero.
print("[2/5] Inicializando la grilla de concentración...")
# Cada celda de la grilla almacenará el valor de concentración del contaminante.
concentration = np.zeros((NY, NX))

# --- 3. EJECUCIÓN DE LA SIMULACIÓN ---
# Este es el corazón del script: un bucle que avanza en el tiempo y calcula el estado
# de la grilla en cada paso usando el método de diferencias finitas.
print("[3/5] Ejecutando la simulación de transporte de contaminantes...")

history = [] # Lista para guardar "fotogramas" de la simulación para el GIF.

# tqdm nos da una barra de progreso para ver el avance de la simulación.
for t in tqdm(range(NT), desc="Simulando propagación"):
    # Guardar el estado actual para la animación a intervalos regulares.
    if t % 10 == 0: # Guardar un frame cada 10 pasos de tiempo.
        history.append(np.copy(concentration))
        
    # Mantener la fuente de contaminación activa durante el período definido.
    if t < SOURCE_DURATION_STEPS:
        concentration[SOURCE_Y, SOURCE_X] = SOURCE_STRENGTH

    # Copiamos la grilla actual para calcular el nuevo estado sin afectar los cálculos del paso actual.
    c_new = concentration.copy()

    # Iterar sobre todas las celdas internas de la grilla para aplicar la ecuación.
    for i in range(1, NY - 1):
        for j in range(1, NX - 1):
            # Aproximación de las derivadas parciales usando diferencias finitas centradas.
            # Término de Advección (transporte por el flujo de agua)
            adv_x = V_X * (concentration[i, j+1] - concentration[i, j-1]) / (2 * DX)
            adv_y = V_Y * (concentration[i+1, j] - concentration[i-1, j]) / (2 * DY)
            
            # Término de Dispersión (expansión y mezcla)
            disp_x = D_L * (concentration[i, j+1] - 2*concentration[i,j] + concentration[i, j-1]) / DX**2
            disp_y = D_T * (concentration[i+1, j] - 2*concentration[i,j] + concentration[i-1, j]) / DY**2
            
            # Aplicar la Ecuación de Advección-Dispersión para calcular el cambio en la concentración.
            change = DT * (disp_x + disp_y - adv_x - adv_y)
            c_new[i, j] = concentration[i, j] + change / POROSITY

    # Actualizar la grilla principal con los nuevos valores calculados.
    concentration = c_new

# --- 4. VISUALIZACIÓN ESTÁTICA FINAL ---
# Creamos una imagen que muestra el estado final de la simulación.
print("[4/5] Creando la imagen del estado final...")
total_years = NT * DT / (365 * 24 * 3600)
fig_static, ax_static = plt.subplots(figsize=(14, 8))
im_static = ax_static.imshow(concentration, cmap='Reds', origin='lower',
                             extent=[0, NX*DX, 0, NY*DY], vmin=0, vmax=SOURCE_STRENGTH * 0.5)
ax_static.set_title(f'Pluma de Contaminación Después de {total_years:.1f} Años', fontsize=16)
ax_static.set_xlabel('Distancia X (m)')
ax_static.set_ylabel('Distancia Y (m)')
cbar_static = fig_static.colorbar(im_static)
cbar_static.set_label('Concentración Relativa')
ax_static.scatter([600, 1000, 1200], [550, 600, 400], c='blue', ec='white', s=100, marker='D', label='Pozos de Monitoreo')
ax_static.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'plume_final_state.png'), dpi=300)
plt.close(fig_static)

# --- 5. CREACIÓN DEL GIF ANIMADO ---
# Usamos los "fotogramas" guardados en la lista `history` para crear una animación.
print("[5/5] Creando el GIF animado...")
fig_anim, ax_anim = plt.subplots(figsize=(14, 8))
ax_anim.set_xlabel('Distancia X (m)')
ax_anim.set_ylabel('Distancia Y (m)')

# Configurar el primer frame de la animación.
im_anim = ax_anim.imshow(history[0], cmap='Reds', origin='lower',
                         extent=[0, NX*DX, 0, NY*DY], vmin=0, vmax=SOURCE_STRENGTH)
cbar_anim = fig_anim.colorbar(im_anim, label='Concentración Relativa')
title = ax_anim.set_title('Día: 0')

# Función de actualización: se llama para cada frame del GIF.
def update(frame):
    """Actualiza los datos de la imagen y el título para el frame actual."""
    im_anim.set_data(history[frame])
    current_days = frame * 10 * (DT / (24*3600)) # 10 es el factor de guardado
    title.set_text(f'Propagación de la Pluma - Día: {current_days:.0f}')
    return [im_anim, title]

# Crear el objeto de animación.
ani = animation.FuncAnimation(fig_anim, update, frames=len(history),
                              interval=50, blit=True) # blit=True optimiza el renderizado.
# Guardar la animación como un archivo GIF.
ani.save(os.path.join(OUTPUT_DIR, 'plume_propagation.gif'), writer='pillow', fps=20)
plt.close(fig_anim)

print("\n¡Éxito! Simulador de pluma de contaminantes completado.")