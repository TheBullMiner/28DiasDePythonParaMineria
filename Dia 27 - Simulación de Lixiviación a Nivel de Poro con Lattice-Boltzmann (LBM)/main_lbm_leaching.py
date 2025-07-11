# ==============================================================================
# #28DiasDePythonParaMineria - Día 27
# Título: Simulación de Lixiviación a Nivel de Poro con Lattice-Boltzmann (LBM)
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script implementa un simulador 2D de la interacción fluido-roca
# utilizando el Método de Lattice-Boltzmann (LBM), una técnica de Dinámica de
# Fluidos Computacional (CFD). Modela cómo un fluido (solución de lixiviación)
# fluye a través de una microestructura porosa y cómo disuelve lentamente el
# mineral, alterando la geometría de los poros en tiempo real.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os
from scipy.ndimage import binary_erosion

# --- 1. CONFIGURACIÓN DEL MODELO LBM Y DE LIXIVIACIÓN ---
print("[1/5] Configurando los parámetros del simulador LBM...")

# Parámetros de la grilla y de la simulación
NX, NY = 200, 100         # Dimensiones de la grilla (ancho, alto)
N_STEPS = 1500            # Pasos de tiempo totales de la simulación
RELAXATION_TIME = 0.8     # Tau (τ): controla la viscosidad cinemática del fluido. Valores cercanos a 0.5 son menos viscosos.
DISSOLUTION_RATE = 0.001  # Tasa de disolución del mineral por contacto con el fluido.

# Crear la carpeta de salida si no existe para evitar errores.
if not os.path.exists('output'): os.makedirs('output')

# --- 2. INICIALIZACIÓN DE LA GRILLA Y EL MEDIO POROSO ---
print("[2/5] Creando la microestructura porosa...")

# Definiciones para LBM D2Q9 (2 Dimensiones, 9 Vectores de Velocidad)
# c: Vectores de velocidad discretos que conectan una celda con sus vecinas.
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]])
# w: Pesos asociados a cada dirección, para calcular las variables macroscópicas.
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Crear una matriz booleana que representa la roca sólida (True) y los poros (False).
np.random.seed(42)
rock_matrix = np.zeros((NY, NX), dtype=bool)
for _ in range(40): # Generar 40 "granos" de mineral de diferentes tamaños aleatoriamente.
    r = np.random.randint(4, 12)
    x, y = np.random.randint(r, NX-r), np.random.randint(r, NY-r)
    Y, X = np.ogrid[:NY, :NX]
    dist = np.sqrt((X - x)**2 + (Y-y)**2)
    rock_matrix[dist <= r] = True

# Grilla que almacena la "densidad" o cantidad de mineral en cada celda sólida.
mineral_grid = np.zeros((NY, NX))
mineral_grid[rock_matrix] = 1.0 # 1.0 significa mineral intacto.

# Inicializar las distribuciones de partículas 'f' en estado de equilibrio con densidad 1 y velocidad 0.
# 'f' es la matriz principal de LBM: [NY, NX, 9], almacena la "cantidad" de fluido que se mueve en cada una de las 9 direcciones.
rho = np.ones((NY, NX))
f = np.zeros((NY, NX, 9))
for i in range(9):
    f[:, :, i] = w[i] * rho

# --- 3. EJECUCIÓN DE LA SIMULACIÓN ---
print(f"[3/5] Simulando {N_STEPS} pasos de tiempo...")
history = [] # Para guardar los frames del GIF

for step in tqdm(range(N_STEPS), desc="Simulando Flujo y Reacción"):
    
    # --- Forzar el flujo con un gradiente de presión ---
    # Se aumenta la densidad en la entrada (izquierda) y se disminuye en la salida (derecha).
    # Esto crea un flujo neto de izquierda a derecha.
    f[:, 0, [1, 5, 8]] = w[[1, 5, 8]] * (1 + 3 * 0.02) # Flujo hacia la derecha
    f[:, -1, [3, 6, 7]] = w[[3, 6, 7]] * (1 - 3 * 0.02) # Flujo saliendo por la derecha
    
    # --- PASO DE PROPAGACIÓN (STREAMING) ---
    # Mover las distribuciones de partículas a las celdas vecinas según su vector de velocidad.
    for i in range(9):
        f[:, :, i] = np.roll(np.roll(f[:, :, i], c[i, 0], axis=1), c[i, 1], axis=0)

    # --- Condición de frontera "Bounce-back" en los obstáculos de roca ---
    # Las partículas que chocan con la roca rebotan, invirtiendo su dirección.
    bounced = f[rock_matrix, :]
    bounced = bounced[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]] # Invertir direcciones [centro, E, N, O, S, NE, NO, SO, SE] -> [centro, O, S, E, N, SO, SE, NO, NE]
    f[rock_matrix, :] = bounced

    # --- PASO DE COLISIÓN ---
    # Las partículas en cada celda interactúan y se redistribuyen.
    # 1. Calcular variables macroscópicas (densidad y velocidad) a partir de 'f'.
    rho = np.sum(f, axis=2)
    ux = np.sum(f * c[:, 0], axis=2) / rho
    uy = np.sum(f * c[:, 1], axis=2) / rho
    
    # 2. Calcular la distribución de equilibrio 'feq' basada en las variables macroscópicas.
    feq = np.zeros_like(f)
    for i in range(9):
        cu = c[i, 0]*ux + c[i, 1]*uy
        feq[:, :, i] = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*(ux**2 + uy**2))
    
    # 3. Relajación hacia el equilibrio.
    f += -(1 / RELAXATION_TIME) * (f - feq)

    # --- PASO DE LIXIVIACIÓN (REACCIÓN FLUIDO-ROCA) ---
    if step > 50: # Empezar a lixiviar después de que el flujo se estabilice un poco.
        # Identificar la superficie de la roca (celdas de roca que tocan poros).
        eroded_rock = binary_erosion(rock_matrix)
        rock_surface = rock_matrix & ~eroded_rock
        
        # Disolver el mineral en la superficie por simple contacto.
        mineral_grid[rock_surface] -= DISSOLUTION_RATE
        mineral_grid = np.clip(mineral_grid, 0, 1.0)
        
        # Si una celda se queda sin mineral, se convierte en un poro.
        rock_matrix[mineral_grid < 0.01] = False
    
    # Guardar el estado actual para la animación.
    if step % 15 == 0:
        fluid_speed = np.sqrt(ux**2 + uy**2)
        # Crear una imagen RGB: Canal Rojo=Mineral, Canal Verde=Velocidad del Fluido.
        frame_data = np.stack([mineral_grid, fluid_speed, np.zeros_like(mineral_grid)], axis=-1)
        history.append(frame_data)

# --- 4. VISUALIZACIÓN ESTÁTICA FINAL ---
print("[4/5] Creando la imagen del estado final...")
fig_static, ax_static = plt.subplots(figsize=(16, 8))
final_frame = history[-1].copy() # Usar una copia.
# Normalizar la velocidad para una mejor visualización.
v_max_final = np.max(final_frame[:,:,1])
if v_max_final > 0:
    final_frame[:,:,1] /= v_max_final
ax_static.imshow(final_frame, origin='lower')
ax_static.set_title(f'Microestructura Final Después de {N_STEPS} Pasos de Tiempo', fontsize=16)
ax_static.set_xticks([]); ax_static.set_yticks([])
plt.savefig('output/lbm_leaching_final.png', dpi=300, bbox_inches='tight')
plt.close(fig_static)

# --- 5. CREACIÓN DEL GIF ANIMADO ---
print("[5/5] Creando el GIF animado...")
fig_anim, ax_anim = plt.subplots(figsize=(16, 8))
ax_anim.set_xticks([]); ax_anim.set_yticks([])

# Preparar el primer frame.
im = ax_anim.imshow(history[0], origin='lower')
title = ax_anim.set_title('Simulación de Lixiviación a Nivel de Poro - Tiempo: 0', fontsize=16)

def update(frame_idx):
    """Función que se llama para cada frame de la animación."""
    frame_data = history[frame_idx].copy()
    # Normalizar el canal de velocidad en cada frame para un color consistente.
    v_max = np.max(frame_data[:, :, 1])
    if v_max > 0:
        frame_data[:, :, 1] /= v_max
    im.set_data(frame_data)
    title.set_text(f'Simulación de Lixiviación a Nivel de Poro - Tiempo: {frame_idx*15}')
    return [im, title]

ani = animation.FuncAnimation(fig_anim, update, frames=len(history), interval=40, blit=True)
ani.save('output/lbm_leaching_animation.gif', writer='pillow', fps=25)
plt.close(fig_anim)

print("\n¡Éxito! Simulador LBM completado.")