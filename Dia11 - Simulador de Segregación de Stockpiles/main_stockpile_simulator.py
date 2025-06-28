# ==============================================================================
# #28DiasDePythonParaMineria - Día 11
# Título: Simulador de Segregación en Stockpiles con Autómatas Celulares
# Autor: Maycol Benavides
#
# Descripción:
# Este script modela el fenómeno físico de la segregación de partículas al
# construir un stockpile cónico. Utiliza un autómata celular 2D para simular
# la caída y el asentamiento de miles de partículas de diferentes tamaños,
# visualizando cómo las partículas gruesas tienden a rodar hacia los bordes
# mientras que las finas se concentran en el núcleo.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --- 1. PARÁMETROS DE LA SIMULACIÓN ---
# Aquí definimos las constantes que controlan nuestro universo simulado.
# Ajustar estos valores permite modelar diferentes tipos de stockpiles y materiales.
print("[1/5] Configurando los parámetros de la simulación...")

# Dimensiones del lienzo de la simulación. Ancho impar para tener un centro perfecto.
GRID_WIDTH = 201
GRID_HEIGHT = 100

# Número total de partículas a simular. Más partículas = stockpile más grande y simulación más larga.
N_PARTICLES = 500000 

# Propiedades del material de alimentación
COARSE_FRACTION = 0.4  # Proporción de partículas gruesas (40%)
COARSE_SIZE = 10.0     # Valor numérico que representa a las partículas gruesas
FINE_SIZE = 1.0        # Valor numérico que representa a las partículas finas
RANDOM_SEED = 42       # Semilla para que la simulación sea reproducible

# --- 2. INICIALIZACIÓN DEL ENTORNO ---
# Preparamos las estructuras de datos (arrays de NumPy) que almacenarán el estado de la simulación.
print("[2/5] Inicializando la grilla del stockpile...")
np.random.seed(RANDOM_SEED)

# Mapa de altura 1D: Almacena la altura actual de la pila en cada columna 'x'.
height_map = np.zeros(GRID_WIDTH, dtype=int)
# Grilla 2D del stockpile: Almacena el tamaño promedio de las partículas en cada celda (y, x).
stockpile_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
# Grilla de conteo: Necesaria para calcular correctamente el tamaño promedio en cada celda.
particle_count = np.zeros((GRID_HEIGHT, GRID_WIDTH))

# --- 3. EJECUCIÓN DE LA SIMULACIÓN (AUTÓMATA CELULAR) ---
# Este es el corazón del script. Iteramos para cada partícula, simulando su caída y asentamiento.
print(f"[3/5] Simulando la caída de {N_PARTICLES:,} partículas...")

# tqdm nos da una bonita barra de progreso para ver el avance.
for particle_num in tqdm(range(N_PARTICLES), desc="Construyendo Stockpile"):
    
    # Decide si la partícula que cae es gruesa o fina, según la fracción definida.
    particle_size = COARSE_SIZE if np.random.rand() < COARSE_FRACTION else FINE_SIZE
    
    # Todas las partículas caen en la columna central para formar un cono.
    px = GRID_WIDTH // 2
    
    # Bucle de "asentamiento": la partícula rueda hasta encontrar un lugar estable.
    while True:
        # Condición de salida de seguridad: si la columna está llena, no podemos añadir más.
        if height_map[px] >= GRID_HEIGHT - 1:
            break

        # Evaluar la altura de la columna actual y sus vecinas inmediatas.
        h_left = height_map[px - 1] if px > 0 else GRID_HEIGHT # Si está en el borde, la vecina es "infinitamente alta".
        h_center = height_map[px]
        h_right = height_map[px + 1] if px < GRID_WIDTH - 1 else GRID_HEIGHT
        
        # --- LÓGICA DE FÍSICA Y SEGREGACIÓN ---
        # Si la columna actual es un mínimo local (más baja o igual que sus vecinas), la partícula se asienta.
        if h_center <= h_left and h_center <= h_right:
            break
        
        # Si no, la partícula es inestable y debe rodar.
        else:
            # Rueda hacia la columna vecina que sea más baja.
            if h_left < h_right:
                px -= 1
            elif h_right < h_left:
                px += 1
            else: # Si ambas son igual de bajas, elige una dirección al azar.
                px += np.random.choice([-1, 1])

        # Se añade un factor de "fricción" para las partículas finas, dándoles una
        # pequeña probabilidad de "pegarse" a una pendiente en lugar de seguir rodando.
        # Esto simula la cohesión y ayuda a formar un núcleo de finos.
        if particle_size == FINE_SIZE and np.random.rand() < 0.2: # 20% de probabilidad de quedarse
            break

        # Condición de salida de seguridad: si la partícula rueda fuera de la grilla.
        if not (0 < px < GRID_WIDTH - 1):
            break
            
    # La partícula ha encontrado su posición final (px).
    py = height_map[px]

    # Otra comprobación de seguridad antes de escribir en los arrays.
    if py >= GRID_HEIGHT: continue
    
    # Actualizar la celda (py, px) con la nueva partícula.
    current_total_size = stockpile_grid[py, px] * particle_count[py, px]
    particle_count[py, px] += 1
    stockpile_grid[py, px] = (current_total_size + particle_size) / particle_count[py, px]
    
    # Incrementar la altura de la columna donde se asentó la partícula.
    height_map[px] += 1

# --- 4. POST-PROCESAMIENTO Y ANÁLISIS ---
# La simulación está completa. Ahora preparamos los datos para una visualización más estética y útil.
print("[4/5] Post-procesando los resultados para la visualización...")

# Crear una copia de la grilla para no modificar la original.
grid_to_smooth = np.copy(stockpile_grid)
# Donde no hay partículas, el valor es 0. Lo convertimos a NaN (Not a Number) para que Matplotlib lo ignore al graficar.
grid_to_smooth[particle_count == 0] = np.nan

# Truco para aplicar un filtro Gaussiano (suavizado) que ignore los valores NaN.
# Esto es esencial para no "difuminar" el stockpile con el fondo vacío.
weights = np.ones(grid_to_smooth.shape)
weights[np.isnan(grid_to_smooth)] = 0
grid_no_nan = np.nan_to_num(grid_to_smooth)

filtered_data = gaussian_filter(grid_no_nan, sigma=2.0)
filtered_weights = gaussian_filter(weights, sigma=2.0)

# Normalizar los datos filtrados por los pesos para corregir los bordes.
filtered_weights[filtered_weights == 0] = 1 
smoothed_grid = filtered_data / filtered_weights
smoothed_grid[weights == 0] = np.nan

# Tomar muestras de la grilla final para crear los histogramas de distribución de tamaño (PSD).
center_sample = smoothed_grid[:, (GRID_WIDTH//2 - 10):(GRID_WIDTH//2 + 10)].flatten()
center_sample = center_sample[~np.isnan(center_sample)]

edge_sample_left = smoothed_grid[:, :50].flatten()
edge_sample_right = smoothed_grid[:, -50:].flatten()
edge_sample = np.concatenate((edge_sample_left, edge_sample_right))
edge_sample = edge_sample[~np.isnan(edge_sample)]


# --- 5. VISUALIZACIÓN ---
# Crear y guardar el panel de gráficos final.
print("[5/5] Creando el gráfico final...")
fig = plt.figure(figsize=(18, 10))
plt.style.use('seaborn-v0_8-whitegrid')
fig.set_facecolor('white')

# Usar GridSpec para un control preciso sobre la disposición de los subplots.
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Panel principal: Heatmap del stockpile.
im = ax1.imshow(smoothed_grid, cmap='coolwarm', aspect='auto', origin='lower',
                interpolation='bilinear', vmin=FINE_SIZE, vmax=COARSE_SIZE)
ax1.set_title('Simulación de Segregación en Stockpile', fontsize=22, weight='bold', pad=20)
ax1.set_xlabel('Posición Horizontal', fontsize=14)
ax1.set_ylabel('Altura', fontsize=14)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label('Tamaño Promedio de Partícula', fontsize=12)

# Panel inferior izquierdo: PSD del centro.
if len(center_sample) > 0:
    ax2.hist(center_sample, bins=20, color='#3B82F6', ec='black', density=True)
else:
    ax2.text(0.5, 0.5, 'Sin datos en esta zona', ha='center', va='center', fontsize=12, color='gray')
ax2.set_title('PSD en el Centro del Stockpile', fontsize=14)
ax2.set_xlabel('Tamaño de Partícula')
ax2.set_ylabel('Densidad')
ax2.set_xlim(0, COARSE_SIZE + 1)

# Panel inferior derecho: PSD del borde.
if len(edge_sample) > 0:
    ax3.hist(edge_sample, bins=20, color='#EF4444', ec='black', density=True)
else:
    ax3.text(0.5, 0.5, 'Sin datos en esta zona', ha='center', va='center', fontsize=12, color='gray')
ax3.set_title('PSD en el Borde del Stockpile', fontsize=14)
ax3.set_xlabel('Tamaño de Partícula')
ax3.set_xlim(0, COARSE_SIZE + 1)

# Ajustar el layout y guardar la figura final.
plt.tight_layout(pad=2.0)
plt.savefig('stockpile_segregation_analysis.png', dpi=300)
print("\n¡Éxito! Gráfico final guardado como 'stockpile_segregation_analysis.png'.")