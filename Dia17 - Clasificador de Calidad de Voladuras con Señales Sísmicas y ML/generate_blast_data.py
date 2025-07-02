import pandas as pd
import numpy as np
# Ya no necesitamos importar ricker de scipy.signal

# --- NUEVA FUNCIÓN PARA REEMPLAZAR A SCIPY.SIGNAL.RICKER ---
def ricker_wavelet(n_points, a):
    """
    Genera una ondícula Ricker (wavelet mexicana).
    
    Args:
        n_points (int): Número de puntos en la wavelet.
        a (float): Parámetro que controla el ancho de la wavelet.
    
    Returns:
        numpy.ndarray: El array que contiene la wavelet.
    """
    t = np.linspace(-a, a, n_points)
    # Fórmula de la wavelet Ricker (segunda derivada de una Gaussiana)
    y = (1.0 - 2.0 * (t**2 / (a/4)**2)) * np.exp(-(t**2 / (2.0 * (a/4)**2)))
    return y
# -------------------------------------------------------------

def generate_blast_signal(quality, duration=1.0, sampling_rate=1000):
    """Genera una forma de onda sísmica simulada."""
    n_points = int(duration * sampling_rate)
    
    if quality == 'Good':
        # Señal más corta, de mayor frecuencia, menos ruido
        # Usamos nuestra nueva función en lugar de ricker()
        main_pulse = 2.0 * ricker_wavelet(n_points // 4, a=4.0)
        signal = np.zeros(n_points)
        start_idx = n_points // 8
        signal[start_idx : start_idx + len(main_pulse)] = main_pulse
        noise = np.random.normal(0, 0.1, n_points)
    else: # 'Bad'
        # Señal más larga, de menor frecuencia, más "arrastrada" y con más ruido
        # Usamos nuestra nueva función en lugar de ricker()
        main_pulse = 1.5 * ricker_wavelet(n_points // 2, a=8.0)
        secondary_pulse = 0.8 * ricker_wavelet(n_points // 3, a=10.0)
        signal = np.zeros(n_points)
        start_idx1 = n_points // 10
        start_idx2 = n_points // 3
        signal[start_idx1 : start_idx1 + len(main_pulse)] = main_pulse
        signal[start_idx2 : start_idx2 + len(secondary_pulse)] += secondary_pulse
        noise = np.random.normal(0, 0.25, n_points)
        
    return signal + noise

# Generar datos para 20 voladuras
print("Generando datos sísmicos simulados...")
all_data = []
for blast_id in range(20):
    quality = 'Good' if blast_id < 10 else 'Bad'
    signal = generate_blast_signal(quality)
    time_steps = np.arange(len(signal)) / 1000.0 # Tiempo en segundos
    
    for t, amp in zip(time_steps, signal):
        all_data.append([blast_id, t, amp, quality])

df = pd.DataFrame(all_data, columns=['blast_id', 'time_s', 'amplitude', 'quality'])
df.to_csv('data/blast_data.csv', index=False)
print("Datos sísmicos simulados creados en 'data/blast_data.csv'")