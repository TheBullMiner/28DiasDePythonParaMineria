# -----------------------------------------------------------------
# ARCHIVO: evaluador_galactus.py (VERSIÓN FINAL COMPLETA)
# DESCRIPCIÓN: Módulo central con TODAS las funciones de cálculo:
#              V12, Rítmica, Multifractal, Wavelet, Péndulos y Densidad de Valor.
# -----------------------------------------------------------------
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree, ConvexHull
from scipy.fft import rfft, rfftfreq
from MFDFA import MFDFA
import pywt

# --- FUNCIÓN BASE V12 ---
def calcular_features_para_punto(punto_coords, k_vecinos, kdtree_global, coords_globales, leyes_globales):
    """
    Calcula un conjunto de características V12 (estadísticas y geométricas)
    para un único punto basándose en sus k vecinos más cercanos.
    """
    if k_vecinos == 0:
        return {'varianza_local': 0.0, 'cv_local': 0.0, 'gradiente_magnitud': 0.0, 'gradiente_x': 0.0, 'gradiente_y': 0.0, 'gradiente_z': 0.0, 'distancia_media_vecinos': 0.0, 'anisotropia_ratio_local': 1.0, 'anisotropia_angulo_local': 0.0}
        
    distancias, indices = kdtree_global.query(punto_coords, k=k_vecinos)
    if k_vecinos == 1:
        distancias, indices = np.array([distancias]), np.array([indices])

    vecinos_coords, vecinos_leyes = coords_globales[indices], leyes_globales[indices]
    
    varianza_local = np.var(vecinos_leyes)
    media_local = np.mean(vecinos_leyes)
    cv_local = np.std(vecinos_leyes) / media_local if media_local > 1e-9 else 0.0

    A = np.c_[vecinos_coords, np.ones(k_vecinos)]
    gradiente_x, gradiente_y, gradiente_z, gradiente_magnitud = 0.0, 0.0, 0.0, 0.0
    try:
        if k_vecinos > 3: # Necesitamos al menos 4 puntos para definir un plano 3D
            coeficientes, _, _, _ = np.linalg.lstsq(A, vecinos_leyes, rcond=None)
            gradiente_x, gradiente_y, gradiente_z = coeficientes[0], coeficientes[1], coeficientes[2]
            gradiente_magnitud = np.sqrt(gradiente_x**2 + gradiente_y**2 + gradiente_z**2)
    except np.linalg.LinAlgError: pass

    distancia_media_vecinos = np.mean(distancias)
    
    anisotropia_ratio, anisotropia_angulo = 1.0, 0.0
    try:
        if k_vecinos > 1:
            pca = PCA(n_components=2).fit(vecinos_coords[:, :2])
            if pca.explained_variance_[1] > 1e-9:
                anisotropia_ratio = pca.explained_variance_[0] / pca.explained_variance_[1]
            componente_principal = pca.components_[0]
            anisotropia_angulo = np.degrees(np.arctan2(componente_principal[1], componente_principal[0]))
            if anisotropia_angulo < 0: anisotropia_angulo += 180
    except Exception: pass

    return {
        'varianza_local': varianza_local, 'cv_local': cv_local, 'gradiente_magnitud': gradiente_magnitud,
        'gradiente_x': gradiente_x, 'gradiente_y': gradiente_y, 'gradiente_z': gradiente_z,
        'distancia_media_vecinos': distancia_media_vecinos, 'anisotropia_ratio_local': anisotropia_ratio,
        'anisotropia_angulo_local': anisotropia_angulo,
    }

# --- FUNCIÓN RÍTMICA ---
def calcular_features_ritmicas(vecinos_coords, vecinos_leyes):
    if len(vecinos_coords) < 4: return {'frecuencia_dominante': 0.0, 'potencia_dominante': 0.0}
    try:
        pca = PCA(n_components=1).fit(vecinos_coords)
        proyecciones = vecinos_coords @ pca.components_[0]
        indices_ordenados, proyecciones_ordenadas = np.argsort(proyecciones), np.sort(proyecciones)
        leyes_ordenadas = vecinos_leyes[indices_ordenados]
        espaciado_medio = np.mean(np.diff(proyecciones_ordenadas))
        if espaciado_medio < 1e-9: return {'frecuencia_dominante': 0.0, 'potencia_dominante': 0.0}
        N = len(leyes_ordenadas)
        yf, xf = rfft(leyes_ordenadas), rfftfreq(N, espaciado_medio)
        idx_max_potencia = np.argmax(np.abs(yf[1:])) + 1
        return {'frecuencia_dominante': xf[idx_max_potencia], 'potencia_dominante': np.abs(yf[idx_max_potencia]) / (N / 2) if N > 0 else 0.0}
    except Exception:
        return {'frecuencia_dominante': 0.0, 'potencia_dominante': 0.0}

# --- FUNCIÓN MULTIFRACTAL ---
def calcular_features_multifractales(vecinos_coords, vecinos_leyes):
    if len(vecinos_coords) < 16 or np.std(vecinos_leyes) < 1e-9: return {'alpha_width': 0.0, 'asymmetry': 0.0}
    try:
        pca = PCA(n_components=1).fit(vecinos_coords)
        proyecciones = vecinos_coords @ pca.components_[0]
        leyes_ordenadas = vecinos_leyes[np.argsort(proyecciones)]
        lag_max = len(leyes_ordenadas) // 4
        if lag_max < 4: return {'alpha_width': 0.0, 'asymmetry': 0.0}
        lag = np.unique(np.logspace(np.log10(4), np.log10(lag_max), 10).astype(int))
        _, dfa = MFDFA(leyes_ordenadas, lag=lag, q=2, order=1)
        alpha, f_alpha = dfa.join()
        alpha_width = np.max(alpha) - np.min(alpha)
        idx_f_max = np.argmax(f_alpha)
        asymmetry = (np.max(alpha) - alpha[idx_f_max]) - (alpha[idx_f_max] - np.min(alpha))
        return {'alpha_width': alpha_width, 'asymmetry': asymmetry}
    except Exception:
        return {'alpha_width': 0.0, 'asymmetry': 0.0}

# --- FUNCIÓN WAVELET ("CUÁNTICA") ---
def calcular_features_wavelet(vecinos_coords, vecinos_leyes, grid_res=(8,8,8), wavelet='db4'):
    if len(vecinos_leyes) < np.prod(grid_res) * 0.2: return {'energia_total_wavelet': 0, 'entropia_wavelet': 0, 'ratio_energia_escala': 0}
    try:
        from scipy.interpolate import griddata
        points, values = vecinos_coords, vecinos_leyes
        grid_x, grid_y, grid_z = (np.linspace(points[:,i].min(), points[:,i].max(), grid_res[i]) for i in range(3))
        gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        grid_points = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
        gridded_data = griddata(points, values, grid_points, method='linear', fill_value=np.mean(values))
        data_cube = gridded_data.reshape(grid_res)
        coeffs = pywt.wavedecn(data_cube, wavelet=wavelet, level=2)
        energia_nivel1 = np.sum([np.sum(c**2) for c in coeffs[1].values()])
        energia_nivel2 = np.sum([np.sum(c**2) for c in coeffs[2].values()])
        energia_total = energia_nivel1 + energia_nivel2
        if energia_total < 1e-9: return {'energia_total_wavelet': 0, 'entropia_wavelet': 0, 'ratio_energia_escala': 0}
        p1, p2 = energia_nivel1 / energia_total, energia_nivel2 / energia_total
        entropia_wavelet = - (p1 * np.log2(p1) if p1 > 0 else 0) - (p2 * np.log2(p2) if p2 > 0 else 0)
        ratio_energia_escala = energia_nivel1 / energia_nivel2 if energia_nivel2 > 1e-9 else energia_nivel1
        return {'energia_total_wavelet': energia_total, 'entropia_wavelet': entropia_wavelet, 'ratio_energia_escala': ratio_energia_escala}
    except Exception:
        return {'energia_total_wavelet': 0, 'entropia_wavelet': 0, 'ratio_energia_escala': 0}

# --- FUNCIÓN DE VETAS CAÓTICAS (PÉNDULO) ---
def trazar_veta_caotica(df_sondajes_subset, n_pasos=10, k_paso=5):
    if len(df_sondajes_subset) < k_paso: return np.array([])
    trayectoria = [df_sondajes_subset[['x', 'y', 'z']].mean().values]
    coords_subset = df_sondajes_subset[['x', 'y', 'z']].values
    kdtree_subset = cKDTree(coords_subset)
    indices_usados = set()
    for _ in range(n_pasos):
        punto_actual = trayectoria[-1]
        _, indices_cercanos = kdtree_subset.query(punto_actual, k=len(coords_subset))
        indices_nuevos = [idx for idx in indices_cercanos if idx not in indices_usados][:k_paso]
        if not indices_nuevos: break
        trayectoria.append(np.mean(coords_subset[indices_nuevos], axis=0))
        indices_usados.update(indices_nuevos)
    return np.array(trayectoria)

def distancia_a_trayectoria(punto, trayectoria):
    if len(trayectoria) == 0: return 9999.0
    return np.min([np.linalg.norm(punto - p_trayectoria) for p_trayectoria in trayectoria])

# --- FUNCIÓN DE DENSIDAD DE VALOR ---
def calcular_feature_potencial_ley(vecinos_coords: np.ndarray, vecinos_leyes: np.ndarray) -> float:
    if len(vecinos_coords) < 4: return 0.0
    try:
        from scipy.spatial import ConvexHull
        volumen = ConvexHull(vecinos_coords).volume
        if volumen < 1e-9: return 1e9
        return np.sum(vecinos_leyes) / volumen
    except Exception:
        return 0.0