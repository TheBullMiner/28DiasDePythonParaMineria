# ==============================================================================
# #28DiasDePythonParaMineria - Día 18
# Título: Análisis 3D de Sobrerotura ("Falso Techo") en Galerías Subterráneas
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script lee una nube de puntos 3D (simulando un escaneo láser) de una
# galería subterránea. Calcula la sobrerotura (overbreak) comparando la
# superficie escaneada con el techo de diseño teórico. Finalmente, genera
# visualizaciones 3D, incluyendo una captura de pantalla y un GIF animado
# "fly-through", para comunicar el estado geotécnico y la dilución.
# ==============================================================================

import pandas as pd
import numpy as np
import pyvista as pv
import imageio
from scipy.interpolate import griddata

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
# Definimos los parámetros de diseño y cargamos los datos del escaneo.
print("[1/5] Cargando la nube de puntos y configurando el diseño...")
df = pd.read_csv('data/tunnel_scan.csv')

# Parámetros del túnel de DISEÑO. Estos definen la forma ideal de la excavación.
TUNNEL_WIDTH = 6.0
TUNNEL_HEIGHT = 5.0

# Crear un objeto PolyData de PyVista, que es una estructura de datos
# fundamental para manejar nubes de puntos y mallas no estructuradas.
cloud = pv.PolyData(df[['x', 'y', 'z']].values)

# --- 2. CÁLCULO DE LA SOBREROTURA (OVERBREAK) ---
# Aquí cuantificamos la desviación entre lo real y lo planeado.
print("[2/5] Calculando la sobrerotura en el techo...")

# Para cada punto en la nube escaneada, calculamos cuál debería haber sido
# su altura Z según el diseño (la ecuación de una elipse/arco).
# np.maximum(0, ...) previene errores de dominio en la raíz cuadrada.
z_design = np.sqrt(np.maximum(0, (TUNNEL_HEIGHT**2) * (1 - (cloud.points[:, 1]**2 / (TUNNEL_WIDTH/2)**2))))

# La sobrerotura es la diferencia vertical entre la altura real y la de diseño.
overbreak = cloud.points[:, 2] - z_design

# Asignamos este nuevo valor como un atributo escalar a nuestra nube de puntos.
# PyVista ahora "sabe" el valor de sobrerotura para cada punto.
cloud['overbreak'] = overbreak

# --- 3. CREACIÓN DE MALLAS 3D (MÉTODO ROBUSTO) ---
# Las nubes de puntos son difíciles de visualizar. Las convertimos en superficies
# (mallas) para un renderizado suave y con colores.
print("[3/5] Creando mallas 3D para la visualización...")

# Usamos interpolación (griddata) para crear una superficie regular a partir de
# la nube de puntos irregular. Este método es muy robusto.
bounds = cloud.bounds
x_res, y_res = 200, 100
x_reg = np.linspace(bounds[0], bounds[1], x_res)
y_reg = np.linspace(bounds[2], bounds[3], y_res)
x_reg, y_reg = np.meshgrid(x_reg, y_reg)

# Interpolar los valores Z (altura) y 'overbreak' en la nueva grilla regular.
z_reg = griddata(cloud.points[:, :2], cloud.points[:, 2], (x_reg, y_reg), method='cubic')
overbreak_reg = griddata(cloud.points[:, :2], overbreak, (x_reg, y_reg), method='cubic')

# Crear la malla de la superficie escaneada (real)
mesh_scan = pv.StructuredGrid(x_reg, y_reg, z_reg)
mesh_scan['overbreak'] = overbreak_reg.flatten('F') # 'F' para orden Fortran

# Crear la malla del techo de DISEÑO (ideal)
z_design_grid = np.sqrt(np.maximum(0, (TUNNEL_HEIGHT**2) * (1 - (y_reg**2 / (TUNNEL_WIDTH/2)**2))))
mesh_design = pv.StructuredGrid(x_reg, y_reg, z_design_grid)

# --- 4. VISUALIZACIÓN ESTÁTICA Y CAPTURA DE PANTALLA ---
# Creamos una vista en planta 2D para un análisis rápido.
print("[4/5] Creando la escena 3D y guardando la captura de pantalla...")

plotter_static = pv.Plotter(off_screen=True, window_size=[1024, 768])
# Añadir la malla escaneada, coloreada por el valor de sobrerotura.
plotter_static.add_mesh(mesh_scan, scalars='overbreak', cmap='coolwarm',
                        scalar_bar_args={'title': 'Sobrerotura (m)'})
# Añadir la malla de diseño como una rejilla para comparar.
plotter_static.add_mesh(mesh_design, style='wireframe', color='black', opacity=0.5)

# Usar métodos automáticos para asegurar que la cámara apunte y se ajuste correctamente.
plotter_static.view_xy()      # Vista en planta (mirando desde arriba)
plotter_static.reset_camera() # Ajusta el zoom para que todo quepa
plotter_static.camera.zoom(1.5)

plotter_static.enable_anti_aliasing() # Para bordes más suaves
screenshot_path = 'output/overbreak_analysis.png'
plotter_static.screenshot(screenshot_path)
print(f"  - Captura de pantalla de la sección guardada en '{screenshot_path}'")
plotter_static.close()

# --- 5. CREACIÓN DEL GIF ANIMADO "FLY-THROUGH" ---
# Creamos la visualización 3D principal y la guardamos como un GIF.
print("[5/5] Creando el GIF animado del 'fly-through'...")

plotter_gif = pv.Plotter(off_screen=True, window_size=[1024, 768])
plotter_gif.add_mesh(mesh_scan, scalars='overbreak', cmap='coolwarm',
                     scalar_bar_args={'title': 'Sobrerotura (m)'})
plotter_gif.add_mesh(mesh_design, style='wireframe', color='black', opacity=0.3)
plotter_gif.enable_anti_aliasing()

# Configurar la vista inicial de la cámara
plotter_gif.view_isometric()
plotter_gif.reset_camera()
plotter_gif.camera.zoom(1.5)

# Flujo de trabajo estándar y robusto para crear un GIF de órbita:
# Paso A: Abrir el archivo GIF para empezar a escribir frames.
gif_path = "output/tunnel_flythrough.gif"
plotter_gif.open_gif(gif_path)

# Paso B: Generar una trayectoria de cámara suave alrededor del objeto.
path = plotter_gif.generate_orbital_path(n_points=60, viewup=[0, 0, 1])

# Paso C: Ejecutar la animación sobre esa trayectoria y escribir los frames.
plotter_gif.orbit_on_path(path, write_frames=True)

# Paso D: Cerrar el plotter para finalizar y guardar el archivo GIF.
plotter_gif.close()

print(f"\n¡Éxito! Visualizaciones 3D generadas en la carpeta 'output'.")