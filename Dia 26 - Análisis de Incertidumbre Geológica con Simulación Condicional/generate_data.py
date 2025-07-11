# Puedes ejecutar esto en una terminal de Python para crear el archivo
import pandas as pd
data = {
    'X': [20, 80, 55, 30, 70, 95, 40, 15],
    'Y': [30, 75, 50, 85, 20, 55, 10, 60],
    'Grade': [1.5, 0.8, 1.2, 0.6, 1.8, 1.1, 1.9, 0.9]
}
df = pd.DataFrame(data)
df.to_csv('data/sample_points.csv', index=False)