# Puedes ejecutar esto en una terminal de Python para crear el archivo
import pandas as pd
import numpy as np
data = {
    'x': [20, 80, 30, 70, 50, 90, 10],
    'y': [30, 70, 80, 20, 50, 50, 60],
    'grade': [1.5, 0.8, 0.6, 1.8, 1.2, 1.1, 0.9]
}
df = pd.DataFrame(data)
df.to_csv('data/drillhole_data.csv', index=False)