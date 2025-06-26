# generate_drillhole_data.py
import pandas as pd
import numpy as np

print("Generando datos de sondaje de prueba...")
# Definir los contactos geológicos
contacts = {
    'OVB': {'to': 40, 'cu_mean': 0.1, 'cu_std': 0.05, 'mo_mean': 10, 'mo_std': 5},
    'DAC': {'to': 70, 'cu_mean': 0.5, 'cu_std': 0.1, 'mo_mean': 400, 'mo_std': 50},
    'POR': {'to': 120, 'cu_mean': 1.6, 'cu_std': 0.4, 'mo_mean': 90, 'mo_std': 25}
}

data = []
current_depth = 0
interval_length = 2

for rock, props in contacts.items():
    while current_depth < props['to']:
        cu = np.random.normal(props['cu_mean'], props['cu_std'])
        mo = np.random.normal(props['mo_mean'], props['mo_std'])
        data.append({
            'from': current_depth,
            'to': current_depth + interval_length,
            'length': interval_length,
            'rock_type': rock,
            'cu_grade': round(abs(cu), 2),
            'mo_ppm': round(abs(mo), 0)
        })
        current_depth += interval_length

df = pd.DataFrame(data)
df.to_csv('data/drillhole_data.csv', index=False)
print("Datos de sondaje guardados en 'data/drillhole_data.csv'")