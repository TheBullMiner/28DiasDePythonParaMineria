import pandas as pd
import numpy as np

def generate_trip(base_profile, n_points, noise_level):
    if base_profile == 'efficient':
        x = np.linspace(0, 10, n_points)
        speed = 10 + 25 * np.sin(x * np.pi / 10)
    elif base_profile == 'aggressive':
        x = np.linspace(0, 10, n_points)
        speed = 40 * np.exp(-((x-2)**2)/2) + 40 * np.exp(-((x-8)**2)*2)
    elif base_profile == 'cautious':
        x = np.linspace(0, 10, n_points)
        speed = 20 * (1 - np.exp(-x*0.5)) - 10 * np.exp(-((x-5)**2)/0.5)
    
    speed += np.random.normal(0, noise_level, n_points)
    return np.clip(speed, 0, 50)

trips = []
trip_id_counter = 0
for profile_name, n_trips in [('efficient', 15), ('aggressive', 10), ('cautious', 12)]:
    for _ in range(n_trips):
        n_points = np.random.randint(80, 120)
        speed_profile = generate_trip(profile_name, n_points, noise_level=1.5)
        for time_step, speed in enumerate(speed_profile):
            trips.append([trip_id_counter, time_step, speed])
        trip_id_counter += 1

df_trips = pd.DataFrame(trips, columns=['trip_id', 'time_step', 'speed'])
df_trips.to_csv('data/truck_trips.csv', index=False)
print("Datos de viaje simulados creados en 'data/truck_trips.csv'")