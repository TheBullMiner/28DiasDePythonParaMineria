# ==============================================================================
# #28DiasDePythonParaMineria - Día 17
# Título: Clasificador de Calidad de Voladuras con Señales Sísmicas y ML
# Autor: Maycol Benavides
# The Bull Miner
#
# Descripción:
# Este script aplica un flujo de trabajo de Machine Learning y Procesamiento de
# Señales para clasificar la calidad de una voladura (ej. "Buena" o "Mala"
# fragmentación) basándose en la forma de onda sísmica que genera.
#
# Flujo de Trabajo:
# 1. Carga datos sísmicos de múltiples voladuras.
# 2. Para cada señal, extrae características cuantitativas (features) como
#    energía, frecuencia dominante, etc.
# 3. Entrena un modelo de clasificación (Random Forest) con estas características
#    y las etiquetas de calidad conocidas.
# 4. Evalúa el rendimiento del modelo.
# 5. Genera un gráfico comparativo de las "firmas sísmicas" de una voladura
#    buena vs. una mala.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- 1. CARGA Y EXTRACCIÓN DE CARACTERÍSTICAS (FEATURE ENGINEERING) ---
# El Machine Learning no funciona con datos crudos (la señal). Necesitamos
# transformar cada señal en un conjunto de números que la describan.
print("[1/5] Cargando datos y extrayendo características de las señales...")

df = pd.read_csv('data/blast_data.csv')
SAMPLING_RATE = 1000 # Frecuencia de muestreo en Hz (puntos por segundo)

features = []
# Agrupamos por 'blast_id' para procesar cada señal de voladura individualmente.
for blast_id, group in df.groupby('blast_id'):
    # Es crucial convertir la serie de Pandas a un array de NumPy con .values
    # para que las librerías científicas como SciPy funcionen correctamente.
    signal = group['amplitude'].values
    
    # Característica 1: Energía Total. Medida simple de la "fuerza" de la señal.
    energy = np.sum(signal**2)
    
    # Usamos el método de Welch para obtener la Densidad Espectral de Potencia (PSD),
    # que nos dice cómo se distribuye la energía de la señal en diferentes frecuencias.
    freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=256)
    
    # Característica 2: Frecuencia Dominante. La frecuencia con la mayor potencia.
    dominant_freq = freqs[np.argmax(psd)]
    
    # Característica 3: Frecuencia Mediana. La frecuencia que divide la energía total en dos mitades iguales.
    median_freq = freqs[np.where(np.cumsum(psd) >= np.sum(psd)/2)[0][0]]
    
    # Característica 4: Duración Significativa. Mide cuánto tiempo dura la parte
    # "fuerte" de la señal, ignorando el ruido inicial y final.
    cumulative_energy = np.cumsum(signal**2)
    total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 0
    if total_energy > 0:
        t5_idx = np.where(cumulative_energy >= 0.05 * total_energy)[0]
        t95_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
        t5 = t5_idx[0] / SAMPLING_RATE if len(t5_idx) > 0 else 0
        t95 = t95_idx[0] / SAMPLING_RATE if len(t95_idx) > 0 else 0
        duration = t95 - t5
    else:
        duration = 0
        
    # Guardamos las características extraídas para esta voladura.
    features.append({
        'blast_id': blast_id, 'energy': energy, 'dominant_freq': dominant_freq,
        'median_freq': median_freq, 'duration': duration, 'quality': group['quality'].iloc[0]
    })

# Convertimos la lista de características en un DataFrame de pandas para facilitar su manejo.
features_df = pd.DataFrame(features)

# --- 2. ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING ---
print("[2/5] Entrenando el clasificador de Machine Learning...")

# Separamos los datos en:
# X: Las características (las variables predictoras).
# y: La etiqueta (la variable que queremos predecir, en este caso 'quality').
X = features_df[['energy', 'dominant_freq', 'median_freq', 'duration']]
y = features_df['quality']

# Dividimos nuestro dataset en un conjunto de entrenamiento (para enseñar al modelo)
# y un conjunto de prueba (para evaluarlo con datos que no ha visto).
# `stratify=y` asegura que la proporción de 'Good' y 'Bad' sea la misma en ambos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Elegimos un clasificador RandomForest, un modelo potente y robusto que funciona bien
# con datos tabulares.
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Entrenamos el modelo con los datos de entrenamiento.
classifier.fit(X_train, y_train)

# --- 3. EVALUACIÓN DEL MODELO ---
print("[3/5] Evaluando el rendimiento del modelo...")

# Usamos el modelo entrenado para hacer predicciones sobre el conjunto de prueba.
y_pred = classifier.predict(X_test)
# Comparamos las predicciones con las etiquetas reales para calcular la precisión.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo en datos de prueba: {accuracy*100:.2f}%")

# La Matriz de Confusión es una herramienta visual excelente para ver qué tipos de errores
# está cometiendo el modelo (ej. ¿confunde 'Good' con 'Bad'?).
cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.title('Matriz de Confusión del Clasificador')
plt.savefig('output/confusion_matrix.png', dpi=150)
plt.close() # Cerramos la figura para que no interfiera con la siguiente.

# --- 4. VISUALIZACIÓN DE "FIRMAS SÍSMICAS" ---
# Para ilustrar el concepto, seleccionamos un ejemplo de una voladura "buena" y una "mala"
# y comparamos sus formas de onda y espectros.
print("[4/5] Creando el gráfico comparativo de firmas sísmicas...")

good_blast_id = features_df[features_df['quality'] == 'Good']['blast_id'].iloc[0]
bad_blast_id = features_df[features_df['quality'] == 'Bad']['blast_id'].iloc[0]

good_signal_data = df[df['blast_id'] == good_blast_id]
bad_signal_data = df[df['blast_id'] == bad_blast_id]

# Creamos un panel de 2x2 para la comparación.
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Análisis Comparativo de Firmas Sísmicas de Voladura', fontsize=22, weight='bold')

# --- Panel para la voladura "Buena" ---
axes[0, 0].plot(good_signal_data['time_s'], good_signal_data['amplitude'], color='#10B981')
axes[0, 0].set_title('Forma de Onda: Voladura de "Buena" Calidad', fontsize=14)
axes[0, 0].set_ylabel('Amplitud')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# Convertimos la Serie de Pandas a un array de NumPy con .values antes de pasarla a welch.
freqs_good, psd_good = welch(good_signal_data['amplitude'].values, fs=SAMPLING_RATE, nperseg=256)
axes[1, 0].plot(freqs_good, psd_good, color='#10B981')
axes[1, 0].set_title('Espectro de Frecuencia (PSD)', fontsize=14)
axes[1, 0].set_xlabel('Frecuencia (Hz)')
axes[1, 0].set_ylabel('Potencia Espectral')
axes[1, 0].set_xlim(0, 150)
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# --- Panel para la voladura "Mala" ---
axes[0, 1].plot(bad_signal_data['time_s'], bad_signal_data['amplitude'], color='#EF4444')
axes[0, 1].set_title('Forma de Onda: Voladura de "Mala" Calidad', fontsize=14)
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

freqs_bad, psd_bad = welch(bad_signal_data['amplitude'].values, fs=SAMPLING_RATE, nperseg=256)
axes[1, 1].plot(freqs_bad, psd_bad, color='#EF4444')
axes[1, 1].set_title('Espectro de Frecuencia (PSD)', fontsize=14)
axes[1, 1].set_xlabel('Frecuencia (Hz)')
axes[1, 1].set_xlim(0, 150)
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

# --- 5. GUARDAR ---
print("[5/5] Guardando el gráfico final...")
plt.tight_layout(rect=[0, 0, 1, 0.95]) # `rect` deja espacio para el supertítulo
plt.savefig('output/blast_signatures.png', dpi=300)
print("\n¡Éxito! Análisis de voladuras completado.")