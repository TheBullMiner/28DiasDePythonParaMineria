# ==============================================================================
# #28DiasDePythonParaMineria - Día 25
# Título: Optimizador de Patrones de Pernos de Roca en 2D
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script genera y visualiza dinámicamente un esquema de fortificación
# para un túnel subterráneo basado en la calidad del macizo rocoso, medida
# por el Índice Q de Barton. El script crea una imagen estática para un caso
# específico y un GIF animado que muestra cómo cambian la longitud y el
# espaciamiento de los pernos a medida que varía la calidad de la roca.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Polygon
import os

# --- 1. CONFIGURACIÓN Y DEFINICIÓN DE LA GEOMETRÍA ---
# Primero, definimos la forma de nuestra excavación y las reglas de ingeniería.
print("[1/4] Definiendo la geometría del túnel y las reglas de diseño...")

# Parámetros para la geometría del túnel (forma de herradura)
TUNNEL_WIDTH = 8.0
TUNNEL_HEIGHT = 6.0
CORNER_RADIUS = 2.0

def create_tunnel_profile(width, height, radius):
    """Crea un polígono de Shapely con forma de túnel herradura."""
    # Puntos de la base y paredes verticales
    p1 = (-width/2, 0)
    p2 = (width/2, 0)
    p3 = (width/2, height - radius)
    p4 = (-width/2, height - radius)
    # Crear el arco del techo con 50 puntos para una curva suave
    arc_points = []
    for angle in np.linspace(0, 180, 50):
        rad = np.deg2rad(angle)
        x = (width/2 - radius) * np.cos(rad)
        y = (height - radius) + (radius * np.sin(rad))
        arc_points.append((x, y))
    # Unir todos los puntos para formar el polígono cerrado
    return Polygon([p1, p2, p3] + arc_points + [p4])

tunnel_poly = create_tunnel_profile(TUNNEL_WIDTH, TUNNEL_HEIGHT, CORNER_RADIUS)

# --- Lógica de Diseño de Fortificación (Simplificada) ---
# Esta función actúa como nuestra "tabla de diseño" o "regla empírica".
# Relaciona un índice de calidad de roca (Q) con un diseño de soporte.
def get_support_design(q_value):
    """Devuelve la longitud y el espaciamiento del perno basado en el valor de Q."""
    if q_value > 10:      # Roca Buena a Muy Buena
        length, spacing = 2.0, 1.8
    elif q_value > 4:     # Roca Regular
        length, spacing = 2.5, 1.5
    elif q_value > 1:     # Roca Mala
        length, spacing = 3.0, 1.2
    elif q_value > 0.1:   # Roca Muy Mala
        length, spacing = 3.5, 1.0
    else:                 # Roca Excepcionalmente Mala
        length, spacing = 4.0, 0.8
    return length, spacing

# Crear la carpeta de salida si no existe
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta '{OUTPUT_DIR}' creada.")

# --- 2. GENERACIÓN DE LA IMAGEN ESTÁTICA ---
# Creamos una imagen para un caso específico para usarla en el post.
print("[2/4] Creando la imagen estática para una calidad de roca 'Mala'...")
Q_STATIC = 2.5
l_static, s_static = get_support_design(Q_STATIC)

fig_static, ax_static = plt.subplots(figsize=(8, 8))
x, y = tunnel_poly.exterior.xy
ax_static.plot(x, y, color='black', linewidth=3)
ax_static.fill(x, y, color='gray', alpha=0.3)

# Lógica para instalar los pernos de forma radial
boundary = tunnel_poly.exterior
total_length = boundary.length
# Calculamos el número de pernos necesarios basados en el espaciamiento
num_bolts = int((total_length * 0.7) / s_static) # Solo fortificamos el 70% superior del perímetro
# Distribuimos los pernos a lo largo del contorno del techo y las paredes superiores
for i in np.linspace(0.15, 0.85, num_bolts):
    point = boundary.interpolate(i, normalized=True)
    # Calcular el vector normal en ese punto para orientar el perno perpendicularmente
    start_point = boundary.interpolate(i - 0.001, normalized=True)
    end_point = boundary.interpolate(i + 0.001, normalized=True)
    dx, dy = end_point.x - start_point.x, end_point.y - start_point.y
    normal = np.array([-dy, dx]) / np.linalg.norm([-dy, dx])
    end_bolt = (point.x - normal[0] * l_static, point.y - normal[1] * l_static)
    ax_static.plot([point.x, end_bolt[0]], [point.y, end_bolt[1]], color='red', lw=2)

ax_static.set_title(f'Diseño de Fortificación para Q = {Q_STATIC}', fontsize=16)
ax_static.set_aspect('equal', adjustable='box')
ax_static.set_xlabel('Metros'); ax_static.set_ylabel('Metros')
plt.savefig(os.path.join(OUTPUT_DIR, 'rock_support_static.png'), dpi=150)
plt.close(fig_static)

# --- 3. CONFIGURACIÓN DEL GIF ANIMADO ---
print("[3/4] Configurando la animación...")
fig_anim, (ax_tunnel, ax_gauge) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [4, 1]})

# Rango de valores de Q a animar, en escala logarítmica que es como se usa Q.
q_values_anim = np.logspace(2, -2, 100) # De 100 (muy buena) a 0.01 (muy mala)
q_labels = ["Excep. Mala", "Muy Mala", "Mala", "Regular", "Buena", "Muy Buena"]
q_ticks = [0.01, 0.1, 1, 4, 10, 40]

# --- 4. CREACIÓN Y GUARDADO DEL GIF ANIMADO ---
print("[4/4] Creando y guardando el GIF animado...")

def animate(i):
    """Función que se llama para dibujar cada frame de la animación."""
    q_val = q_values_anim[i]
    bolt_length, bolt_spacing = get_support_design(q_val)
    
    # --- Dibujar el panel del túnel ---
    ax_tunnel.clear()
    x, y = tunnel_poly.exterior.xy
    ax_tunnel.plot(x, y, color='black', linewidth=3)
    ax_tunnel.fill(x, y, color='gray', alpha=0.3)
    
    # Dibujar los pernos para el valor de Q actual
    boundary = tunnel_poly.exterior
    num_bolts = int((boundary.length * 0.7) / bolt_spacing)
    for j in np.linspace(0.15, 0.85, num_bolts):
        point = boundary.interpolate(j, normalized=True)
        start_point = boundary.interpolate(j - 0.001, normalized=True)
        end_point = boundary.interpolate(j + 0.001, normalized=True)
        dx, dy = end_point.x - start_point.x, end_point.y - start_point.y
        normal = np.array([-dy, dx]) / np.linalg.norm([-dy, dx])
        end_bolt = (point.x - normal[0] * bolt_length, point.y - normal[1] * bolt_length)
        ax_tunnel.plot([point.x, end_bolt[0]], [point.y, end_bolt[1]], color='red', lw=2)
        
    ax_tunnel.set_title('Diseño de Fortificación Dinámico', fontsize=16)
    ax_tunnel.set_aspect('equal', adjustable='box')
    ax_tunnel.set_xlim(-TUNNEL_WIDTH, TUNNEL_WIDTH); ax_tunnel.set_ylim(-1, TUNNEL_HEIGHT + 1)

    # --- Dibujar el panel del "medidor" de calidad de roca ---
    ax_gauge.clear()
    ax_gauge.set_xscale('log')
    ax_gauge.set_xlim(0.01, 100); ax_gauge.set_ylim(0, 1)
    ax_gauge.set_yticks([])
    ax_gauge.set_xticks(q_ticks)
    ax_gauge.set_xticklabels(q_labels, rotation=45, ha='right')
    ax_gauge.set_xlabel('Índice de Calidad de Roca (Q)')
    
    # Dibujar la aguja del medidor que apunta al valor de Q actual
    ax_gauge.arrow(q_val, 1, 0, -0.6, head_width=q_val*0.2, head_length=0.2, fc='blue', ec='blue', width=q_val*0.05)
    ax_gauge.text(q_val, 0.2, f'Q = {q_val:.2f}', ha='center', fontsize=12, weight='bold')

ani = animation.FuncAnimation(fig_anim, animate, frames=len(q_values_anim), interval=100)
ani.save(os.path.join(OUTPUT_DIR, 'rock_support_animation.gif'), writer='pillow', fps=15)
plt.close(fig_anim)

print("\n¡Éxito! Visualizaciones de diseño de fortificación generadas.")