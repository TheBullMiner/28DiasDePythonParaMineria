#Genera datos de prueba para el modelo de bloques y el diseño de voladura
#Se corre primero, luego el main.
# generate_test_data.py
import pandas as pd
import numpy as np

# Generar Modelo de Bloques
print("Generando modelo de bloques de prueba...")
x = np.arange(0, 101, 10)
y = np.arange(0, 101, 10)
z = np.arange(80, 101, 5)
xx, yy, zz = np.meshgrid(x, y, z)
bm = pd.DataFrame({'x': xx.flatten(), 'y': yy.flatten(), 'z': zz.flatten()})
bm['dx'] = bm['dy'] = 10
bm['dz'] = 5
bm['density'] = 2.7

# Crear un cuerpo mineralizado inclinado
bm['is_ore'] = bm['z'] < (-0.2 * bm['x'] - 0.1 * bm['y'] + 105)
bm['rock_type'] = np.where(bm['is_ore'], 'Ore', 'Waste')
bm['cu_grade'] = np.where(bm['is_ore'], np.random.normal(1.2, 0.3), np.random.normal(0.1, 0.05))
bm['cu_grade'] = bm['cu_grade'].clip(0)
bm.to_csv('data/block_model.csv', index=False)
print("Modelo de bloques guardado en 'data/block_model.csv'")

# Generar Diseño de Voladura
print("Generando diseño de voladura de prueba...")
x_blast = np.arange(40, 71, 10)
y_blast = np.arange(40, 71, 10)
xx_b, yy_b = np.meshgrid(x_blast, y_blast)
blast_design = pd.DataFrame({'x': xx_b.flatten(), 'y': yy_b.flatten()})
blast_design['hole_id'] = [f"BH-{i+1:02d}" for i in range(len(blast_design))]
blast_design['z_collar'] = 100
blast_design['depth'] = 15
blast_design = blast_design[['hole_id', 'x', 'y', 'z_collar', 'depth']]
blast_design.to_csv('data/blast_design.csv', index=False)
print("Diseño de voladura guardado en 'data/blast_design.csv'")