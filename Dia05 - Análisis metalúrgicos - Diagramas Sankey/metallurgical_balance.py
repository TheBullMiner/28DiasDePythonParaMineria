#The Bull Miner
#Maycol Benavides
#Dia05 Python para Minería


import plotly.graph_objects as go
import pandas as pd

# --- 1. DATOS DE ENTRADA Y CÁLCULOS METALÚRGICOS ---
print("[1/3] Definiendo datos de entrada y calculando el balance...")

feed_grade = 1.15
conc_grade = 28.5
tail_grade = 0.12

recovery = (conc_grade * (feed_grade - tail_grade)) / (feed_grade * (conc_grade - tail_grade)) * 100
mass_pull = (feed_grade - tail_grade) / (conc_grade - tail_grade) * 100

print("\n--- Resultados del Balance Metalúrgico ---")
print(f"Recuperación de Cobre: {recovery:.2f}%")
print(f"Rendimiento Másico (Mass Pull): {mass_pull:.2f}%")

# --- 2. PREPARACIÓN DE DATOS PARA LOS DIAGRAMAS SANKEY ---
print("\n[2/3] Preparando datos para la visualización de flujos...")

nodes = ['Alimentación (Feed)', 'Concentrado (Conc)', 'Colas (Tails)']

# --- Datos para el Sankey de MASA TOTAL ---
feed_mass = 100
conc_mass = feed_mass * (mass_pull / 100)
tail_mass = feed_mass - conc_mass
source_mass = [0, 0]
target_mass = [1, 2]
value_mass = [conc_mass, tail_mass]

# --- Datos para el Sankey de METAL ---
feed_metal = feed_mass * (feed_grade / 100)
conc_metal = conc_mass * (conc_grade / 100)
tail_metal = tail_mass * (tail_grade / 100)
source_metal = [0, 0]
target_metal = [1, 2]
value_metal = [conc_metal, tail_metal]

# --- 3. GENERACIÓN Y EXPORTACIÓN DE LOS GRÁFICOS (VERSIÓN MEJORADA) ---
print("[3/3] Generando y guardando los diagramas de Sankey mejorados...")

def create_sankey_diagram_v2(source, target, value, title, unit):
    """Función mejorada para crear un diagrama de Sankey estéticamente superior."""
    
    # Preparamos las etiquetas para los nodos y los flujos
    node_labels = [
        f"Alimentación<br>{value[0] + value[1]:.2f} {unit}",  # Feed
        f"Concentrado<br>{value[0]:.2f} {unit}",             # Conc
        f"Colas<br>{value[1]:.2f} {unit}"                    # Tail
    ]
    
    # Definimos el layout del Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # 'snap' fuerza los nodos a las coordenadas X dadas
        node=dict(
            pad=25,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=node_labels,
            # --- LA MAGIA ESTÁ AQUÍ ---
            # Coordenadas X e Y para cada nodo (en un rango de 0 a 1)
            x=[0.01, 0.99, 0.99],  # Feed a la izquierda, Conc y Tails a la derecha
            y=[0.5, 0.2, 0.8],     # Feed en el medio, Conc arriba, Tails abajo
            # -------------------------
            color=["#3B82F6", "#16A34A", "#EF4444"] # Azul, Verde, Rojo
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            # Añadimos etiquetas a los flujos con porcentajes relativos
            label=[
                f"{value[0]:.2f} {unit} ({value[0]*100/(value[0]+value[1]):.1f}%)", 
                f"{value[1]:.2f} {unit} ({value[1]*100/(value[0]+value[1]):.1f}%)"
            ],
            color="rgba(150, 150, 150, 0.5)",
            arrowlen=15 # Añadimos una pequeña flecha para indicar la dirección
        ))])

    fig.update_layout(
        title_text=f"<b>{title}</b>",
        font_family="Arial",
        font_size=14,
        title_font_size=22,
        title_x=0.5,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

# Generar y guardar el gráfico de Masa 
fig_mass = create_sankey_diagram_v2(source_mass, target_mass, value_mass,
                                    "Balance de Masa del Circuito (Base 100t)", "Toneladas")
fig_mass.write_html("output_mass_flow_v2.html")

# Generar y guardar el gráfico de Metal 
# Para el flujo de metal, la etiqueta de recuperación es más importante que el %
metal_labels = [
    f"Concentrado<br>{value_metal[0]:.2f} t Cu<br><b>Recuperación: {recovery:.2f}%</b>",
    f"Colas<br>{value_metal[1]:.2f} t Cu<br>Pérdida: {100-recovery:.2f}%"
]
metal_node_labels = [f"Alimentación<br>{feed_metal:.2f} t Cu"] + metal_labels

fig_metal = create_sankey_diagram_v2(source_metal, target_metal, value_metal,
                                     "Balance de Metal (Cobre Fino)", "t Cu")

# Sobreescribimos las etiquetas de los nodos para el gráfico de metal para añadir la recuperación
fig_metal.data[0].node.label = [f"Alimentación<br>{feed_metal:.2f} t Cu", metal_labels[0], metal_labels[1]]

fig_metal.write_html("output_metal_flow_v2.html")


print("\n¡Éxito! Gráficos mejorados guardados como '..._v2.html'.")