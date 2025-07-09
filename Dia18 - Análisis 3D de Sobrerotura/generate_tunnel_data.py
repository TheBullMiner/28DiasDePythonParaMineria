import numpy as np
import pandas as pd

print("Generando nube de puntos de túnel simulada...")

# Parámetros del túnel de diseño
TUNNEL_WIDTH = 6.0
TUNNEL_HEIGHT = 5.0
TUNNEL_LENGTH = 100.0
N_POINTS = 20000

# Crear la forma base del túnel (un arco)
y = np.random.uniform(-TUNNEL_WIDTH / 2, TUNNEL_WIDTH / 2, N_POINTS)
z_base = np.sqrt((TUNNEL_HEIGHT**2) * (1 - (y**2 / (TUNNEL_WIDTH/2)**2)))

# Crear los puntos a lo largo del túnel
x = np.random.uniform(0, TUNNEL_LENGTH, N_POINTS)

# Simular la sobrerotura (overbreak) en el techo
# Haremos una "panza" grande en el medio del túnel
overbreak_factor = 1 + 0.8 * np.exp(-((x - TUNNEL_LENGTH / 2)**2) / (2 * 20**2))
z = z_base * overbreak_factor
z += np.random.normal(0, 0.05, N_POINTS) # Añadir algo de ruido

# Crear el DataFrame
df = pd.DataFrame({'x': x, 'y': y, 'z': z})
df.to_csv('data/tunnel_scan.csv', index=False)

print("Datos de escaneo de túnel guardados en 'data/tunnel_scan.csv'")