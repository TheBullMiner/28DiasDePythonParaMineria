import pandas as pd
import matplotlib.pyplot as plt
import jinja2
from weasyprint import HTML, CSS
import json
import base64
from io import BytesIO
import os

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
print("[1/5] Cargando datos...")
DATA_PATH = 'data'
TEMPLATE_PATH = 'templates'
OUTPUT_PATH = 'output'

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Cargar la información general del turno
with open(os.path.join(DATA_PATH, 'shift_info.json'), 'r') as f:
    shift_info = json.load(f)

# Cargar datos de FMS y Perforación
df_fms = pd.read_csv(os.path.join(DATA_PATH, 'fms_data.csv'))
df_drill = pd.read_csv(os.path.join(DATA_PATH, 'drill_data.csv'))

# --- 2. PROCESAMIENTO Y CÁLCULO DE KPIs ---
print("[2/5] Calculando KPIs...")

# KPIs de Acarreo
mineral_tons = df_fms[df_fms['destination'] == 'Planta']['payload_tons'].sum()
waste_tons = df_fms[df_fms['destination'] == 'Botadero']['payload_tons'].sum()
prod_by_truck = df_fms[df_fms['destination'] == 'Planta'].groupby('truck_id')['payload_tons'].sum()

# KPIs de Perforación
drilled_meters = df_drill['depth_meters'].sum()
drill_summary = df_drill.groupby('drill_id').agg(
    total_meters=('depth_meters', 'sum'),
    hole_count=('hole_id', 'count')
).reset_index()

# Consolidar KPIs en un diccionario
kpis = {
    "total_mineral_tons": mineral_tons,
    "total_waste_tons": waste_tons,
    "total_drilled_meters": drilled_meters,
    "drill_summary": drill_summary.to_dict(orient='records')
}

# --- 3. GENERACIÓN DEL GRÁFICO ---
print("[3/5] Creando visualizaciones...")

fig, ax = plt.subplots(figsize=(8, 4))
prod_by_truck.sort_values().plot(kind='barh', ax=ax, color='#0055A4')
ax.set_title('Producción de Mineral por Camión')
ax.set_xlabel('Toneladas')
ax.set_ylabel('Camión')
plt.tight_layout()

# Convertir el gráfico a una imagen en memoria para embeber en el PDF
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_png = buffer.getvalue()
buffer.close()

# Codificar la imagen en Base64
grafico_produccion_b64 = base64.b64encode(image_png).decode('utf-8')

# --- 4. RENDERIZADO DE LA PLANTILLA HTML ---
print("[4/5] Ensamblando el reporte...")

# Configurar el entorno de Jinja2
env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_PATH))
template = env.get_template('template.html')

# Preparar el contexto con todos los datos para la plantilla
context = {
    "fecha": shift_info['fecha'],
    "turno": shift_info['turno'],
    "supervisor": shift_info['supervisor'],
    "kpis": kpis,
    "grafico_produccion_b64": grafico_produccion_b64
}

# Renderizar el HTML con los datos
html_out = template.render(context)

# --- 5. GENERACIÓN DEL PDF ---
print("[5/5] Exportando a PDF...")

# Cargar el CSS
css = CSS(os.path.join(TEMPLATE_PATH, 'style.css'))

# Crear el PDF
pdf_filename = f"reporte_turno_{shift_info['fecha']}_{shift_info['turno'].split(' ')[1]}.pdf"
pdf_path = os.path.join(OUTPUT_PATH, pdf_filename)
HTML(string=html_out, base_url=TEMPLATE_PATH).write_pdf(pdf_path, stylesheets=[css])

print(f"\n¡Éxito! Reporte generado en: {pdf_path}")