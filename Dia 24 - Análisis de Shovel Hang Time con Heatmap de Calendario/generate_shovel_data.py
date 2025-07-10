import pandas as pd
import numpy as np

def generate_day_data(day):
    """Genera datos de un día de operación."""
    events = []
    current_time = day
    end_time = day + pd.Timedelta(hours=24)
    
    while current_time < end_time:
        # Ciclo de carguío
        dig_time = np.random.uniform(15, 25)
        events.append({'timestamp': current_time, 'status': 'Digging'})
        current_time += pd.Timedelta(seconds=dig_time)
        
        swing_time = np.random.uniform(10, 20)
        events.append({'timestamp': current_time, 'status': 'Swinging'})
        current_time += pd.Timedelta(seconds=swing_time)
        
        dump_time = np.random.uniform(10, 15)
        events.append({'timestamp': current_time, 'status': 'Dumping'})
        current_time += pd.Timedelta(seconds=dump_time)
        
        # ¿Hay un camión esperando?
        # Hacemos que los fines de semana (días 5 y 6) tengan más esperas
        if day.weekday() in [5, 6]:
            prob_no_truck = 0.4
        else:
            prob_no_truck = 0.2
            
        if np.random.rand() < prob_no_truck:
            wait_time = np.random.exponential(scale=120) # Esperas más largas en promedio
            events.append({'timestamp': current_time, 'status': 'Waiting for Truck'})
            current_time += pd.Timedelta(seconds=wait_time)
            
    return events

# Generar datos para un mes
print("Generando datos de estado de la pala para un mes...")
start_date = '2023-10-01'
date_range = pd.to_datetime(pd.date_range(start_date, periods=31, freq='D'))
all_events = []
for day in date_range:
    all_events.extend(generate_day_data(day))

df = pd.DataFrame(all_events)
df.to_csv('data/shovel_status.csv', index=False)
print("Datos simulados guardados en 'data/shovel_status.csv'")