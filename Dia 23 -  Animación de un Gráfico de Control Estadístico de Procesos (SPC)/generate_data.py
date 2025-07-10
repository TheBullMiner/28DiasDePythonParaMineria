# Puedes ejecutar esto en una terminal de Python para crear el archivo
import pandas as pd
import numpy as np

np.random.seed(42)
# Proceso estable
stable_process = np.random.normal(loc=28.5, scale=0.5, size=80)
# Proceso con una desviación (la media sube)
shift_process = np.random.normal(loc=30.5, scale=0.6, size=20)
# Un punto atípico extremo
stable_process[50] = 26.0

# Combinar y crear DataFrame
data = np.concatenate([stable_process, shift_process])
df = pd.DataFrame({'hour': range(1, 101), 'concentrate_grade': data})
df.to_csv('data/process_data.csv', index=False)
print("Datos de proceso simulados creados en 'data/process_data.csv'")