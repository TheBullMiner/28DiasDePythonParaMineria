# ==============================================================================
# #28DiasDePythonParaMineria - Día 21
# Título: Simulador de Lixiviación en Pila con Autómatas Celulares
# Autor: Maycol Benavides
# The Bull Miner
# 
# Descripción:
# Este script modela el proceso hidrometalúrgico de lixiviación en pila (heap
# leaching) en una sección transversal 2D. Utiliza un modelo de autómata
# celular para simular dos fenómenos clave: el descenso de un frente de
# solución y la cinética de disolución del metal. El resultado es una
# visualización animada del proceso de recuperación a lo largo del tiempo.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os

# --- 1. PARÁMETROS DE LA SIMULACIÓN Y DE LA PILA ---
# Aquí definimos las constantes que controlan nuestro universo simulado.
# Ajustar estos valores permite modelar diferentes tipos de pilas y minerales.
print("[1/5] Configurando los parámetros de la simulación...")

# Parámetros de la grilla (nuestro "universo") y del tiempo
GRID_WIDTH = 150       # Ancho de la pila en celdas
GRID_HEIGHT = 75       # Altura de la pila en celdas
NT = 400               # Número total de pasos de tiempo a simular (ej. días)
RANDOM_SEED = 42       # Para que la simulación sea reproducible

# Parámetros del proceso físico y químico
INITIAL_METAL_PER_CELL = 100.0  # Unidades de metal lixiviable en cada celda al inicio
LEACH_KINETIC_CONSTANT_K = 0.05 # Constante de cinética: un valor más alto significa lixiviación más rápida
WETTING_FRONT_SPEED = 0.9       # Velocidad de descenso del frente húmedo (celdas por día)

# --- Crear la carpeta de salida si no existe ---
# Es una buena práctica para evitar errores de 'FileNotFoundError'.
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta '{OUTPUT_DIR}' creada.")

# --- 2. INICIALIZACIÓN DEL ENTORNO ---
# Preparamos los arrays de NumPy que almacenarán el estado de la simulación.
print("[2/5] Construyendo la pila de lixiviación virtual...")
np.random.seed(RANDOM_SEED)

# Grilla 2D principal: almacena el metal remanente en cada celda.
heap_grid = np.full((GRID_HEIGHT, GRID_WIDTH), INITIAL_METAL_PER_CELL)
# Grilla auxiliar para llevar la cuenta de cuánto tiempo ha estado "húmeda" cada celda (no usada en esta versión, pero útil para modelos más complejos).
time_wet_grid = np.zeros_like(heap_grid, dtype=int)
# Array 1D que representa la posición vertical del frente de solución en cada columna.
wetting_front_z = np.full(GRID_WIDTH, GRID_HEIGHT, dtype=float)

# Calculamos el metal total al inicio para poder calcular la recuperación.
TOTAL_INITIAL_METAL = np.sum(heap_grid)

# --- 3. EJECUCIÓN DE LA SIMULACIÓN ---
# Este es el corazón del script: un bucle que avanza en el tiempo y actualiza el
# estado de la pila en cada paso.
print(f"[3/5] Simulando el ciclo de lixiviación de {NT} días...")

# Listas para guardar "fotogramas" de la simulación para el GIF.
heap_history = []
recovery_history = []

# tqdm nos da una bonita barra de progreso.
for t in tqdm(range(NT), desc="Lixiviando la Pila"):
    # Guardar el estado actual para la animación a intervalos regulares.
    if t % 4 == 0: # Guardar un frame cada 4 pasos de tiempo para una animación más fluida.
        heap_history.append(np.copy(heap_grid))
    
    # Simular el avance del frente húmedo.
    # Añadimos un poco de aleatoriedad para simular canalizaciones y flujos no uniformes.
    wetting_front_z -= WETTING_FRONT_SPEED * np.random.uniform(0.8, 1.2, size=GRID_WIDTH)
    wetting_front_z = np.clip(wetting_front_z, 0, GRID_HEIGHT) # No puede bajar de 0.

    # Crear una "máscara" booleana: un array de True/False que nos dice qué celdas están "mojadas".
    y_indices, x_indices = np.indices(heap_grid.shape)
    is_wet_mask = y_indices >= wetting_front_z[x_indices]
    
    # Incrementar el contador de tiempo húmedo.
    time_wet_grid[is_wet_mask] += 1
    
    # --- Aplicar la Cinética de Lixiviación (Modelo de Primer Orden Simplificado) ---
    # La cantidad de metal que se disuelve en este paso de tiempo es proporcional al metal que aún queda.
    leached_amount = LEACH_KINETIC_CONSTANT_K * heap_grid
    
    # La lixiviación solo ocurre en las celdas que están húmedas (debajo del frente).
    heap_grid[is_wet_mask] -= leached_amount[is_wet_mask]
    heap_grid = np.clip(heap_grid, 0, INITIAL_METAL_PER_CELL) # Asegurarse de que el metal no sea negativo.
    
    # Calcular la recuperación acumulada en este paso de tiempo.
    current_total_metal = np.sum(heap_grid)
    recovery = (1 - current_total_metal / TOTAL_INITIAL_METAL) * 100
    recovery_history.append(recovery)

# --- 4. VISUALIZACIÓN ESTÁTICA FINAL ---
# Creamos una imagen que muestra el estado final de la pila.
print("[4/5] Creando la imagen del estado final...")
fig_static, ax_static = plt.subplots(figsize=(12, 7))
im_static = ax_static.imshow(heap_grid, cmap='hot_r', origin='lower', vmin=0, vmax=INITIAL_METAL_PER_CELL)
ax_static.set_title(f'Metal Remanente en la Pila Después de {NT} Días', fontsize=16)
ax_static.set_xlabel('Posición Horizontal')
ax_static.set_ylabel('Altura en la Pila')
cbar = fig_static.colorbar(im_static)
cbar.set_label('Unidades de Metal Remanente')
plt.savefig(os.path.join(OUTPUT_DIR, 'heap_leach_final_state.png'), dpi=300)
plt.close(fig_static)

# --- 5. CREACIÓN DEL GIF ANIMADO ---
# Usamos los "fotogramas" guardados para crear la animación.
print("[5/5] Creando el GIF animado...")
fig_anim = plt.figure(figsize=(16, 9))
# Usamos GridSpec para un layout de múltiples gráficos más complejo y controlado.
gs = GridSpec(3, 1, height_ratios=[0.15, 2.5, 1])
ax_title = fig_anim.add_subplot(gs[0])
ax_heap = fig_anim.add_subplot(gs[1])
ax_recovery = fig_anim.add_subplot(gs[2])

# Configurar el título dinámico
ax_title.axis('off')
title = ax_title.text(0.5, 0.5, '', ha='center', va='center', fontsize=20, weight='bold')

# Configurar el heatmap de la pila
im = ax_heap.imshow(heap_history[0], cmap='hot_r', origin='lower', vmin=0, vmax=INITIAL_METAL_PER_CELL)
ax_heap.set_ylabel('Altura en la Pila')
ax_heap.set_xticks([])

# Configurar el gráfico de la curva de recuperación
line, = ax_recovery.plot([], [], color='#10B981', lw=3)
ax_recovery.set_xlim(0, NT)
ax_recovery.set_ylim(0, 100)
ax_recovery.set_xlabel('Días de Lixiviación')
ax_recovery.set_ylabel('Recuperación Acumulada (%)')
ax_recovery.grid(True, linestyle='--')

# Función de actualización: se llama para cada frame del GIF.
def update(frame):
    day = frame * 4 # Calcular el día actual
    im.set_data(heap_history[frame]) # Actualizar el heatmap
    title.set_text(f'Simulación de Lixiviación en Pila - Día {day}') # Actualizar el título
    line.set_data(range(day + 1), recovery_history[:day + 1]) # Actualizar la curva
    return [im, line, title]

# Crear y guardar la animación.
ani = animation.FuncAnimation(fig_anim, update, frames=len(heap_history), interval=50, blit=True)
ani.save(os.path.join(OUTPUT_DIR, 'heap_leach_simulation.gif'), writer='pillow', fps=20)
plt.close(fig_anim)

print("\n¡Éxito! Simulador de lixiviación completado.")