# ==============================================================================
# #28DiasDePythonParaMineria - Día 16
# Título: Generador de Isosuperficies de Ley a partir de un Modelo de Bloques
# Autor: Maycol Benavides
# The Bull Miner
# Descripción:
# Este script lee un modelo de bloques en formato CSV, lo reconstruye como una
# grilla 3D y utiliza el algoritmo Marching Cubes para generar y visualizar
# mallas 3D (isosuperficies) para diferentes leyes de corte. El resultado
# es una visualización 3D anidada del cuerpo mineralizado, guardada como una
# imagen estática y un GIF animado.
# ==============================================================================

import pandas as pd
import numpy as np
import pyvista as pv
import imageio # Importar imageio es una buena práctica aunque pyvista lo llame por debajo

# --- 1. CARGA DE DATOS Y PREPARACIÓN DE LA GRILLA 3D ---
# El primer paso es cargar los datos planos (CSV) y darles una estructura 3D
# que una librería de visualización como PyVista pueda entender.
print("[1/4] Cargando el modelo de bloques...")

# Cargar los datos desde el CSV.
# Se asume que el CSV está ordenado por Z, luego por Y, y finalmente por X.
df = pd.read_csv('data/block_model_3d.csv')

# Crear una estructura de PyVista (StructuredGrid) a partir de los datos.
# Este objeto entiende la topología 3D del modelo de bloques.
print("      - Reconstruyendo la grilla 3D...")
# Determinar las dimensiones de la grilla (nº de bloques en cada eje)
nx = len(df['x'].unique())
ny = len(df['y'].unique())
nz = len(df['z'].unique())

# Crear el objeto de la grilla y asignarle las dimensiones
grid = pv.StructuredGrid()
grid.dimensions = (nx, ny, nz)

# Asignar las coordenadas de los puntos (centroides de los bloques)
grid.points = df[['x', 'y', 'z']].values

# Añadir la ley como un atributo escalar a los puntos de la grilla.
# Este es el valor que usaremos para el contorneado.
grid['cu_grade'] = df['cu_grade'].values

# --- 2. GENERACIÓN Y VISUALIZACIÓN DE ISOSUPERFICIES ---
# Aquí creamos las mallas 3D (sólidos) para cada ley de corte y las
# añadimos a una escena virtual.
print("[2/4] Generando y añadiendo las isosuperficies a la escena...")

# Definir las leyes de corte (cut-offs) y su apariencia visual
cutoffs = [0.5, 1.0, 1.5]
colors = ['#3B82F6', '#F97316', '#DC2626'] # Azul, Naranja, Rojo
opacities = [0.2, 0.5, 0.9] # La de menor ley será más transparente

# Crear el objeto Plotter de PyVista. Es nuestro "lienzo" o "escenario" 3D.
# `off_screen=True` es crucial para que el script corra en un servidor o
# sin abrir una ventana emergente.
plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])

# Bucle para generar y añadir cada isosuperficie individualmente.
# Este enfoque es más robusto que intentar añadirlas todas de una vez.
for cutoff, color, opacity in zip(cutoffs, colors, opacities):
    print(f"      - Generando superficie para ley de corte: {cutoff}%")
    
    # El método `contour` aplica el algoritmo Marching Cubes para el cutoff actual.
    surface = grid.contour(isosurfaces=[cutoff], scalars='cu_grade')
    
    # Añadir ESTA malla específica al plotter con SU estilo específico.
    # Es importante comprobar si se generó alguna geometría (`n_points > 0`).
    if surface.n_points > 0:
        plotter.add_mesh(
            surface,
            color=color,
            opacity=opacity,
            smooth_shading=True # Mejora la apariencia visual de la iluminación
        )

# --- 3. CONFIGURACIÓN FINAL DE LA ESCENA Y CAPTURA ---
# Añadimos elementos de contexto y configuramos la cámara para una buena vista.
print("[3/4] Configurando la escena 3D y guardando capturas...")

# Añadir elementos de contexto a la escena
plotter.add_bounding_box(color='grey', line_width=2) # Dibuja un cubo delimitador
plotter.show_grid() # Muestra los ejes y una grilla en el plano base

# Configurar la cámara para una buena vista inicial
plotter.camera_position = 'iso' # Vista isométrica
plotter.camera.zoom(1.2) # Acercar un poco la cámara

# Activar anti-aliasing para un renderizado más suave y de mayor calidad
plotter.enable_anti_aliasing()

# Guardar una captura de pantalla estática del estado actual de la cámara
screenshot_path = 'output/orebody_isosurfaces.png'
plotter.screenshot(screenshot_path)
print(f"  - Captura de pantalla guardada en '{screenshot_path}'")


# --- 4. CREACIÓN DEL GIF ANIMADO ---
# Usamos un flujo de trabajo compatible con versiones recientes de PyVista
# para crear una animación de la cámara orbitando alrededor del objeto.
print("[4/4] Creando GIF animado de la rotación...")

# Paso A: Definir la ruta del archivo y abrir el "writer" del GIF.
# El plotter ahora sabe que cualquier acción de renderizado debe ir a este archivo.
gif_path = "output/orebody_isosurfaces.gif"
plotter.open_gif(gif_path)

# Paso B: Generar una trayectoria suave para la cámara.
# `generate_orbital_path` crea una secuencia de posiciones de cámara.
path = plotter.generate_orbital_path(n_points=60, viewup=[0, 0, 1])

# Paso C: Ejecutar la animación a lo largo de la trayectoria.
# `write_frames=True` le dice al plotter que guarde cada paso de la animación
# como un frame en el archivo GIF que abrimos en el Paso A.
plotter.orbit_on_path(path=path, write_frames=True, step=0.05)

# Paso D: Cerrar el plotter y el archivo GIF para finalizar el proceso y guardar.
# Esto es crucial para asegurarse de que el archivo no quede corrupto.
plotter.close()

print(f"\n¡Éxito! Visualizaciones 3D generadas en la carpeta 'output'.")