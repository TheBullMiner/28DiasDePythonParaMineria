# ==============================================================================
# #28DiasDePythonParaMineria - Día 13
# Título: Simulador de Colas en la Chancadora con SimPy
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script utiliza la simulación de eventos discretos para modelar y analizar
# la formación de colas de camiones en una chancadora. Modela la llegada de
# camiones y el tiempo de servicio como procesos estocásticos (aleatorios)
# para obtener métricas realistas de congestión.
#
# Técnica Clave:
# - Simulación de Eventos Discretos (DES): Un paradigma de modelado donde el
#   estado del sistema solo cambia en puntos discretos en el tiempo.
# - Librería SimPy: Un framework potente y flexible en Python para construir
#   simulaciones de eventos discretos.
# - Teoría de Colas (M/M/1): El modelo subyacente que estamos simulando, donde
#   las llegadas son de tipo Markoviano (Poisson) y el servicio también (Exponencial),
#   con un solo servidor (la chancadora).
# ==============================================================================

import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. PARÁMETROS DE LA SIMULACIÓN ---
# Definimos las constantes que gobernarán nuestro modelo.
# Cambiar estos valores permite simular diferentes escenarios operativos.
print("[1/4] Configurando los parámetros de la simulación...")

RANDOM_SEED = 42         # Semilla para que los resultados aleatorios sean reproducibles.
SIM_TIME = 8 * 60        # Tiempo total de simulación en minutos (un turno de 8 horas).

# Parámetros del sistema (el 'lambda' y 'mu' de la teoría de colas), en minutos.
# El tiempo entre llegadas sigue una distribución exponencial, lo que equivale
# a que el número de llegadas en un intervalo sigue una distribución de Poisson.
TRUCK_ARRIVAL_RATE = 5.0   # En promedio, un camión llega cada 5 minutos.

# El tiempo de servicio también sigue una distribución exponencial.
CRUSHER_SERVICE_RATE = 4.5 # En promedio, la chancadora tarda 4.5 minutos en atender un camión.

# Listas globales para recolectar datos durante la simulación para su posterior análisis.
wait_times = []
queue_length_timeline = []
time_points = []

# --- 2. DEFINICIÓN DE LOS PROCESOS DE SIMULACIÓN ---
# En SimPy, el mundo está compuesto por procesos. Aquí definimos los nuestros.
print("[2/4] Definiendo los procesos de la simulación...")

class CrusherQueue:
    """Representa el sistema de la chancadora."""
    def __init__(self, env, service_rate):
        self.env = env
        # El concepto clave: un 'Resource' de SimPy. Representa un recurso limitado
        # que los procesos deben solicitar. Aquí, solo hay 1 chancadora.
        self.crusher = simpy.Resource(env, capacity=1)
        self.service_rate = service_rate

    def serve(self, truck_id):
        """Generador que simula el proceso de descarga de un camión."""
        # 'yield' es la palabra clave en SimPy. 'timeout' pausa este proceso
        # por un tiempo determinado, permitiendo que otros procesos se ejecuten.
        service_time = np.random.exponential(self.service_rate)
        yield self.env.timeout(service_time)

def truck(env, truck_id, crusher_queue):
    """Generador que representa el ciclo de vida de un camión que llega a la chancadora."""
    arrival_time = env.now # Registrar el tiempo de llegada. `env.now` es el reloj de la simulación.
    
    # 'with crusher_queue.crusher.request() as request:' es la forma de solicitar el recurso.
    # El proceso se pausará aquí (`yield request`) hasta que el recurso (chancadora) esté libre.
    with crusher_queue.crusher.request() as request:
        yield request
        
        # Una vez que el 'yield' termina, significa que el camión ha obtenido el recurso.
        # Calculamos y guardamos el tiempo que tuvo que esperar.
        wait_time = env.now - arrival_time
        wait_times.append(wait_time)
        
        # Ahora que el camión está en la chancadora, llamamos al proceso de servicio.
        # El proceso del camión se pausará hasta que el proceso de servicio termine.
        yield env.process(crusher_queue.serve(truck_id))

def setup(env, arrival_rate, service_rate):
    """Generador principal que crea nuevos camiones a lo largo del tiempo."""
    crusher_queue = CrusherQueue(env, service_rate)
    
    # Bucle infinito para generar llegadas de camiones durante toda la simulación.
    i = 0
    while True:
        # Pausar por un tiempo exponencialmente distribuido antes de crear el siguiente camión.
        yield env.timeout(np.random.exponential(arrival_rate))
        i += 1
        # Iniciar el proceso para el nuevo camión.
        env.process(truck(env, f'Camión-{i}', crusher_queue))
        
        # Inmediatamente después de que llega un camión, registramos el estado de la cola.
        # `crusher_queue.crusher.queue` es la lista de procesos esperando.
        # `crusher_queue.crusher.count` es el número de procesos usando el recurso (0 o 1).
        q_len = len(crusher_queue.crusher.queue) + crusher_queue.crusher.count
        queue_length_timeline.append(q_len)
        time_points.append(env.now)

# --- 3. EJECUCIÓN DE LA SIMULACIÓN ---
# Aquí es donde todo se pone en marcha.
print("[3/4] Ejecutando la simulación de eventos discretos...")
np.random.seed(RANDOM_SEED) # Asegurar reproducibilidad.
env = simpy.Environment() # Crear el entorno de simulación.
env.process(setup(env, TRUCK_ARRIVAL_RATE, CRUSHER_SERVICE_RATE)) # Añadir el proceso de configuración inicial.
env.run(until=SIM_TIME) # Correr la simulación hasta el tiempo límite.

# --- 4. ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS ---
# La simulación ha terminado. Ahora, analizamos los datos recolectados.
print("[4/4] Analizando resultados y creando el gráfico...")

# Calcular los KPIs clave a partir de las listas de resultados.
avg_wait_time = np.mean(wait_times)
max_wait_time = np.max(wait_times)
pct_wait_over_5_min = (np.sum(np.array(wait_times) > 5) / len(wait_times)) * 100
avg_queue_length = np.mean(queue_length_timeline)

# Imprimir un resumen en la consola.
print("\n--- Resultados de la Simulación ---")
print(f"Tiempo Promedio de Espera en Cola: {avg_wait_time:.2f} minutos")
print(f"Tiempo Máximo de Espera en Cola: {max_wait_time:.2f} minutos")
print(f"Longitud Promedio de la Cola: {avg_queue_length:.2f} camiones")
print(f"% de Camiones que Esperan > 5 min: {pct_wait_over_5_min:.2f}%")

# Creación del panel de gráficos final.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})
plt.style.use('seaborn-v0_8-whitegrid')
fig.suptitle('Análisis de Congestión en la Chancadora (Simulación de Colas)', fontsize=20, weight='bold')

# Gráfico Superior: Longitud de la Cola vs. Tiempo.
# `drawstyle='steps-post'` es ideal para mostrar cómo la cola cambia en puntos discretos.
ax1.plot(time_points, queue_length_timeline, drawstyle='steps-post', color='#3B82F6')
ax1.axhline(y=avg_queue_length, color='red', linestyle='--', label=f'Longitud Promedio ({avg_queue_length:.2f})')
ax1.set_title('Evolución de la Longitud de la Cola en el Tiempo')
ax1.set_xlabel('Tiempo (Minutos)')
ax1.set_ylabel('Nº de Camiones en el Sistema')
ax1.legend()
ax1.grid(True)

# Gráfico Inferior: Histograma de Tiempos de Espera.
# `density=True` normaliza el histograma para que el área total sea 1, representando una distribución de probabilidad.
ax2.hist(wait_times, bins=30, color='#10B981', ec='black', density=True)
ax2.axvline(x=avg_wait_time, color='red', linestyle='--', label=f'Espera Promedio ({avg_wait_time:.2f} min)')
ax2.set_title('Distribución de los Tiempos de Espera en Cola')
ax2.set_xlabel('Tiempo de Espera (Minutos)')
ax2.set_ylabel('Densidad de Probabilidad')
ax2.legend()
ax2.grid(True)

# Ajustar y guardar la figura.
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('output/crusher_queue_analysis.png', dpi=300)

print("\n¡Éxito! Gráfico de análisis de colas guardado como 'output/crusher_queue_analysis.png'.")