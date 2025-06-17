# --------------------------------------------------------------------------
# #28DiasDePythonParaMineria - Día 02 
# Título: Optimización de Mezclas con Restricciones Realistas (Costo y Calidad)
# Autor: Maycol Benavides
# Fecha: 17/06/2025
#
# Descripción:
# Este script resuelve un problema de optimización de mezclas con Modelo de Programación Lineal. 
#
# Librerías necesarias:
# pip install pulp matplotlib
# --------------------------------------------------------------------------

import pulp
import matplotlib.pyplot as plt

# --- PASO 1: Definición de los Datos del Problema ---

stockpiles = {
    'Stockpile_A': {'tons_available': 20000, 'cost_per_ton': 15, 'cu_grade': 0.8, 'as_grade': 0.05},
    'Stockpile_B': {'tons_available': 15000, 'cost_per_ton': 12, 'cu_grade': 1.2, 'as_grade': 0.15},
    'Stockpile_C': {'tons_available': 30000, 'cost_per_ton': 10, 'cu_grade': 0.5, 'as_grade': 0.02},
    'Stockpile_D': {'tons_available': 10000, 'cost_per_ton': 18, 'cu_grade': 1.5, 'as_grade': 0.25},
}

planta_req = {
    'total_feed_tons': 10000,
    'max_as_grade_plant': 0.10,
    'max_total_cost': 125000  # <<< LA RESTRICCIÓN CLAVE QUE CAMBIA EL RESULTADO
}

# --- PASO 2: Creación del Modelo de Programación Lineal ---

model = pulp.LpProblem("Optimizacion_Mezcla_Costo_Restringido", pulp.LpMaximize)

tons_to_take = pulp.LpVariable.dicts(
    "Tons", stockpiles.keys(), lowBound=0, cat='Continuous'
)

# --- PASO 3: Definición del Objetivo y las Restricciones ---

model += pulp.lpSum([tons_to_take[s] * stockpiles[s]['cu_grade'] for s in stockpiles]), "Total_Cobre_Maximizar"

model += pulp.lpSum([tons_to_take[s] for s in stockpiles]) == planta_req['total_feed_tons'], "Alimentacion_Exacta_Planta"

model += pulp.lpSum([
    tons_to_take[s] * stockpiles[s]['as_grade'] for s in stockpiles
]) <= planta_req['max_as_grade_plant'] * planta_req['total_feed_tons'], "Maximo_Arsenico_Permitido"

model += pulp.lpSum([
    tons_to_take[s] * stockpiles[s]['cost_per_ton'] for s in stockpiles
]) <= planta_req['max_total_cost'], "Maximo_Costo_Total"

for s in stockpiles:
    model += tons_to_take[s] <= stockpiles[s]['tons_available'], f"Maximo_Tons_{s}"

# --- PASO 4: Resolver el Problema ---

model.solve(pulp.PULP_CBC_CMD(msg=False))

# --- PASO 5: Función para Generar el Gráfico ---

def generar_grafico_resultados(resultados_dict, titulo, filename):
    nombres_stockpiles = list(resultados_dict.keys())
    toneladas_extraer = list(resultados_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#004c6d', '#4d7c9a', '#8bafc7', '#c8e3f6']
    bars = ax.bar(nombres_stockpiles, toneladas_extraer, color=colors)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{yval:,.0f} T', ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_ylabel('Toneladas a Extraer', fontsize=12)
    ax.set_title(titulo, fontsize=16, weight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, max(toneladas_extraer) * 1.25)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[+] Gráfico de resultados guardado como '{filename}'")

# --- PASO 6: Presentación de Resultados en Consola y Llamada al Gráfico ---

print("="*60)
print(f"Día 02: Resultados de Optimización con Restricción de Costo")
print("="*60)
print(f"Estado de la solución: {pulp.LpStatus[model.status]}")

if pulp.LpStatus[model.status] == 'Optimal':
    total_tons_mezcla = 0
    total_costo_mezcla = 0
    total_cobre_fino = 0
    total_arsenico_fino = 0

    print("\n--- Plan de Extracción Óptimo ---")
    datos_para_grafico = {s: tons_to_take[s].varValue for s in stockpiles}
    
    for s, tons in datos_para_grafico.items():
        print(f"  - Extraer {tons:,.2f} toneladas de {s}")
        total_tons_mezcla += tons
        total_costo_mezcla += tons * stockpiles[s]['cost_per_ton']
        total_cobre_fino += tons * stockpiles[s]['cu_grade']
        total_arsenico_fino += tons * stockpiles[s]['as_grade']

    ley_cu_final = total_cobre_fino / total_tons_mezcla
    ley_as_final = total_arsenico_fino / total_tons_mezcla
    costo_promedio_ton = total_costo_mezcla / total_tons_mezcla

    print("\n--- Resumen de la Mezcla Final ---")
    print(f"Tonelaje Total Alimentado: {total_tons_mezcla:,.2f} Toneladas")
    print(f"Costo Total de Extracción: ${total_costo_mezcla:,.2f} (Límite <= ${planta_req['max_total_cost']:,})")
    print(f"Ley de Cobre (Cu) en Cabeza: {ley_cu_final:.3f}% (Objetivo Maximizado)")
    print(f"Ley de Arsénico (As) en Cabeza: {ley_as_final:.3f}% (Límite <= {planta_req['max_as_grade_plant']:.3f}%)")
    
    titulo_grafico = f'Plan Óptimo con Presupuesto de ${planta_req["max_total_cost"]:,}'
    generar_grafico_resultados(datos_para_grafico, titulo_grafico, "dia02_plan_costo_restringido.png")

else:
    print("\nNo se encontró una solución óptima.")

print("="*60)