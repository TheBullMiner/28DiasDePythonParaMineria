# ==============================================================================
# #28DiasDePythonParaMineria - Día 22
# Título: Simulador de Molienda SAG/AG con Modelo de Matriz de Rotura
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script modela el proceso de conminución dentro de un molino utilizando
# un modelo de balance de población de primer orden. Simula cómo una
# distribución de tamaños de partícula (PSD) de alimentación evoluciona
# con el tiempo para alcanzar un estado estacionario, considerando la rotura
# de partículas y la descarga selectiva por tamaño.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

# --- 1. CONFIGURACIÓN DEL MODELO Y PARÁMETROS ---
# Aquí definimos las constantes que controlan nuestro molino virtual y el material.
print("[1/5] Configurando los parámetros del molino y del material...")

# Definición de las clases de tamaño usando una serie de malla √2, estándar en la industria.
N_CLASSES = 12
MAX_SIZE_mm = 128
sizes = np.array([MAX_SIZE_mm / (2**(i/2)) for i in range(N_CLASSES)])
size_labels = [f"{s:.1f}" for s in sizes] # Etiquetas para los gráficos

# Parámetros de la simulación
N_STEPS = 200  # Pasos de tiempo (ej. segundos o minutos)
FEED_RATE = 100 # Toneladas de material fresco que entran por paso de tiempo
INITIAL_CHARGE_TONS = 1000 # Carga inicial de material dentro del molino

# --- Cinética de Molienda (el "corazón" de la física del modelo) ---

# Tasa de rotura específica (k), también conocida como Función de Selección.
# Define qué tan rápido se rompen las partículas de cada tamaño.
# Típicamente, las partículas más grandes se rompen más rápido (ley de potencias).
k_ref = 0.8 # Tasa de rotura para el tamaño de referencia
size_ref = sizes[0]
alpha = 1.5 # Exponente que controla la dependencia del tamaño
breakage_rate_k = k_ref * (sizes / size_ref)**alpha

# Función de descarga (d), también conocida como Función de Clasificación.
# Define la probabilidad de que una partícula de un tamaño dado salga del molino.
# Típicamente, las partículas más finas salen más fácilmente.
d_ref = 0.5
beta = -2.0 # Exponente negativo para que los finos salgan más
discharge_rate_d = d_ref * (sizes / size_ref)**beta
discharge_rate_d = np.clip(discharge_rate_d, 0, 1) # La probabilidad no puede ser > 1.

# --- Crear la carpeta de salida si no existe ---
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta '{OUTPUT_DIR}' creada.")

# --- 2. CONSTRUCCIÓN DE LA MATRIZ DE ROTURA (B) ---
# La Matriz de Distribución de Rotura (B) es fundamental.
# B_ij define qué fracción de masa de la clase de tamaño j reporta a la clase i
# cuando el material de la clase j se rompe.
print("[2/5] Construyendo la Matriz de Rotura (B)...")
B = np.zeros((N_CLASSES, N_CLASSES))
for j in range(N_CLASSES):
    # La rotura solo puede producir partículas más pequeñas, por eso i > j.
    for i in range(j + 1, N_CLASSES):
        # Usamos un modelo simple de distribución triangular para el ejemplo.
        B[i, j] = 2 * (sizes[i-1] - sizes[i]) / sizes[j]**2 * sizes[i]

# Normalizar cada columna para que la suma de los fragmentos sea igual a la masa rota (100%).
col_sums = B.sum(axis=0)
col_sums[col_sums == 0] = 1 # Evitar la división por cero en la última columna (la más fina).
B = B / col_sums

# --- 3. EJECUCIÓN DE LA SIMULACIÓN DE BALANCE DE POBLACIÓN ---
print(f"[3/5] Simulando {N_STEPS} pasos de tiempo...")

# Definir la PSD de la alimentación (Feed) - ej. una distribución normal centrada en los gruesos.
feed_psd = np.exp(-((np.arange(N_CLASSES) - 2)**2) / (2*2**2))
feed_psd /= feed_psd.sum() # Normalizar para que sume 1.
feed_per_step = feed_psd * FEED_RATE # Flujo másico de alimentación por clase de tamaño.

# El estado inicial del molino es una carga con la misma PSD que la alimentación.
mill_content = feed_psd * INITIAL_CHARGE_TONS

# Listas para guardar el historial para la animación.
mill_history = []
product_history = []

for _ in tqdm(range(N_STEPS), desc="Moliendo"):
    # Guardar el estado actual para la animación.
    mill_history.append(np.copy(mill_content))
    
    # --- Aplicar la Ecuación de Balance de Población ---
    # 1. Material que se rompe y DESAPARECE de su clase de tamaño original.
    breakage_disappearance = breakage_rate_k * mill_content
    
    # 2. Material que APARECE en clases de tamaño más finas debido a la rotura.
    # Esta es la multiplicación de matrices clave: la matriz B distribuye la masa rota.
    breakage_appearance = B @ breakage_disappearance
    
    # 3. Material que sale del molino por la descarga.
    discharge_mass = discharge_rate_d * mill_content
    product_history.append(discharge_mass)
    
    # 4. Actualizar el contenido del molino (Masa Balance: Entradas - Salidas).
    mill_content += (feed_per_step - discharge_mass - breakage_disappearance + breakage_appearance)
    mill_content = np.clip(mill_content, 0, None) # No puede haber masa negativa.

# --- 4. VISUALIZACIÓN ESTÁTICA FINAL ---
print("[4/5] Creando la imagen estática final...")
final_mill_psd = mill_content / mill_content.sum()
final_product_psd = product_history[-1] / product_history[-1].sum()

fig_static, ax_static = plt.subplots(figsize=(12, 8))
ax_static.plot(sizes, feed_psd, 'o-', label='Alimentación (Feed) PSD', color='blue')
ax_static.plot(sizes, final_mill_psd, 's-', label=f'Contenido del Molino (t={N_STEPS}) PSD', color='red')
ax_static.plot(sizes, final_product_psd, '^-', label='Producto (Product) PSD', color='green')

ax_static.set_xscale('log') # El eje X de un PSD siempre es logarítmico.
ax_static.invert_xaxis()    # Por convención, los gruesos van a la izquierda.
ax_static.set_title('Distribución de Tamaño de Partícula (PSD) del Circuito', fontsize=16)
ax_static.set_xlabel('Tamaño de Partícula (mm)')
ax_static.set_ylabel('Fracción de Masa (%)')
ax_static.legend()
ax_static.grid(True, which="both", ls="--")
plt.savefig(os.path.join(OUTPUT_DIR, 'mill_psd_summary.png'), dpi=300)
plt.close(fig_static)

# --- 5. CREACIÓN DEL GIF ANIMADO ---
print("[5/5] Creando el GIF animado...")
fig_anim, ax_anim = plt.subplots(figsize=(12, 7))

def animate(frame):
    """Función que se llama para cada frame de la animación."""
    ax_anim.clear() # Limpiar el frame anterior.
    
    # Normalizar el contenido del molino en ese frame para mostrarlo como una distribución.
    current_psd = mill_history[frame] / mill_history[frame].sum()
    
    # Dibujar el histograma.
    ax_anim.bar(range(N_CLASSES), current_psd, color='#10B981', ec='black')
    ax_anim.set_xticks(range(N_CLASSES))
    ax_anim.set_xticklabels(size_labels, rotation=45, ha='right')
    
    ax_anim.set_title(f'Evolución del Contenido del Molino - Tiempo: {frame}', fontsize=16)
    ax_anim.set_xlabel('Tamaño de Partícula (mm)')
    ax_anim.set_ylabel('Fracción de Masa')
    ax_anim.set_ylim(0, np.max(feed_psd) * 1.1) # Fijar el eje Y para una mejor comparación visual.

ani = animation.FuncAnimation(fig_anim, animate, frames=len(mill_history), interval=50)
ani.save(os.path.join(OUTPUT_DIR, 'mill_content_evolution.gif'), writer='pillow', fps=15)
plt.close(fig_anim)

print("\n¡Éxito! Simulador de molienda completado.")