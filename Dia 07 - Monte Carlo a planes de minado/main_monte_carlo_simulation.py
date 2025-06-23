#Maycol Benavides
# Monte Carlo Simulation for Mining Project Cash Flow Analysis
#The Bull Miner
#Día 07 de python para minería

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm

# --- 1. CONFIGURACIÓN DE PARÁMETROS Y DISTRIBUCIONES ---
print("[1/4] Configurando los parámetros del modelo...")

# Número de iteraciones para la simulación
N_SIMULATIONS = 10000

# Parámetros del plan minero definidos como distribuciones de probabilidad
# En lugar de un solo número, definimos la media y la desviación estándar.
# Esto representa nuestra incertidumbre sobre cada variable.

# Tonelaje (distribución normal)
tons_mean = 5_000_000  # 5 millones de toneladas
tons_std = 300_000     # +/- 300,000 toneladas de incertidumbre

# Ley de Cobre (distribución normal)
grade_mean = 1.2  # % Cu
grade_std = 0.15  # +/- 0.15% de incertidumbre

# Recuperación Metalúrgica (distribución normal)
recovery_mean = 88.0  # %
recovery_std = 2.5    # +/- 2.5% de incertidumbre

# Precio del Cobre (distribución log-normal, más realista para precios)
# Usaremos la media y desviación del logaritmo del precio
# Supongamos un precio esperado de $4.0/lb, lo convertimos a $/tonelada (1 ton = 2204.62 lbs)
price_per_lb_mean = 4.0
price_per_ton_mean = price_per_lb_mean * 2204.62
price_std_dev_factor = 0.2 # Factor de volatilidad
price_log_std = np.log(1 + price_std_dev_factor)
price_log_mean = np.log(price_per_ton_mean) - 0.5 * price_log_std**2

# Costo Operacional (distribución normal)
cost_per_ton_mean = 25.0  # $/tonelada
cost_per_ton_std = 2.0    # +/- $2/t de incertidumbre

# --- 2. EJECUCIÓN DE LA SIMULACIÓN DE MONTE CARLO ---
print(f"[2/4] Ejecutando simulación de Monte Carlo con {N_SIMULATIONS:,} iteraciones...")

# Generar muestras aleatorias para cada variable
sim_tons = norm.rvs(loc=tons_mean, scale=tons_std, size=N_SIMULATIONS)
sim_grade = norm.rvs(loc=grade_mean, scale=grade_std, size=N_SIMULATIONS)
sim_recovery = norm.rvs(loc=recovery_mean, scale=recovery_std, size=N_SIMULATIONS)
sim_price = lognorm.rvs(s=price_log_std, scale=np.exp(price_log_mean), size=N_SIMULATIONS)
sim_cost = norm.rvs(loc=cost_per_ton_mean, scale=cost_per_ton_std, size=N_SIMULATIONS)

# --- 3. CÁLCULO DEL RESULTADO (FLUJO DE CAJA) ---
print("[3/4] Calculando el flujo de caja para cada escenario...")

# Calcular el Cobre Fino Producido para cada iteración
# (Tonelaje * Ley/100 * Recuperación/100)
sim_copper_produced = sim_tons * (sim_grade / 100) * (sim_recovery / 100)

# Calcular Ingresos y Costos Totales
sim_revenue = sim_copper_produced * sim_price
sim_total_cost = sim_tons * sim_cost

# Calcular el Flujo de Caja (Cash Flow)
sim_cash_flow = sim_revenue - sim_total_cost

# --- 4. ANÁLISIS DE RESULTADOS Y VISUALIZACIÓN ---
print("[4/4] Analizando resultados y generando el gráfico...")

# Calcular los percentiles clave (P10, P50, P90)
p10 = np.percentile(sim_cash_flow, 10)
p50 = np.percentile(sim_cash_flow, 50)
p90 = np.percentile(sim_cash_flow, 90)

# Imprimir resumen
print("\n--- Resultados de la Simulación ---")
print(f"Flujo de Caja Esperado (Media): ${np.mean(sim_cash_flow):,.0f}")
print(f"P90 (Caso Pesimista): ${p90:,.0f} (90% de prob. de ser mayor a este valor)")
print(f"P50 (Mediana/Caso Base): ${p50:,.0f}")
print(f"P10 (Caso Optimista): ${p10:,.0f} (10% de prob. de ser mayor a este valor)")

# Creación del gráfico
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Histograma y Curva de Densidad
sns.histplot(sim_cash_flow, bins=50, kde=True, ax=ax,
             color='skyblue', line_kws={'linewidth': 2})

# Líneas verticales para los percentiles
ax.axvline(p10, color='g', linestyle='--', linewidth=2, label=f'P10 (Optimista): ${p10/1e6:.1f}M')
ax.axvline(p50, color='b', linestyle='-', linewidth=2, label=f'P50 (Mediana): ${p50/1e6:.1f}M')
ax.axvline(p90, color='r', linestyle='--', linewidth=2, label=f'P90 (Pesimista): ${p90/1e6:.1f}M')

# Estilo del gráfico
ax.set_title('Distribución de Probabilidad del Flujo de Caja Anual', fontsize=20, weight='bold')
ax.set_xlabel('Flujo de Caja ($)', fontsize=14)
ax.set_ylabel('Frecuencia (Densidad de Probabilidad)', fontsize=14)
ax.legend(fontsize=12)

# Formatear el eje X para que sea más legible (en millones)
formatter = plt.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M')
ax.xaxis.set_major_formatter(formatter)

plt.tight_layout()

# Guardar el gráfico
output_filename = 'cash_flow_distribution.png'
plt.savefig(output_filename, dpi=300)

print(f"\n¡Éxito! Gráfico de distribución guardado como '{output_filename}'.")