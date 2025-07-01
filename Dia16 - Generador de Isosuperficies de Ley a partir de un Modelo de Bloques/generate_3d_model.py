import pandas as pd
import numpy as np

print("Generando modelo de bloques 3D de prueba...")

# Dimensiones de la grilla
nx, ny, nz = 30, 30, 30
x = np.linspace(0, 290, nx)
y = np.linspace(0, 290, ny)
z = np.linspace(0, 290, nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Crear una distribución de ley gaussiana en el centro
center_x, center_y, center_z = 145, 145, 145
sigma_x, sigma_y, sigma_z = 60, 80, 40

# Calcular la ley basada en una distribución gaussiana 3D
exponent = -(((xx - center_x)**2 / (2 * sigma_x**2)) +
             ((yy - center_y)**2 / (2 * sigma_y**2)) +
             ((zz - center_z)**2 / (2 * sigma_z**2)))

grade = 2.0 * np.exp(exponent) + np.random.normal(0, 0.05, xx.shape)
grade = np.clip(grade, 0, 2.0)

# Crear DataFrame
bm = pd.DataFrame({
    'x': xx.flatten(),
    'y': yy.flatten(),
    'z': zz.flatten(),
    'cu_grade': grade.flatten()
})

bm.to_csv('data/block_model_3d.csv', index=False)
print("Modelo de bloques guardado en 'data/block_model_3d.csv'")