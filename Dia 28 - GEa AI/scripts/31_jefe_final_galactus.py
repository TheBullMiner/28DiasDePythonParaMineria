# -----------------------------------------------------------------
# ARCHIVO: 31_jefe_final_galactus.py (VERSIÓN FINAL COMPLETA)
# DESCRIPCIÓN: ¡El Devorador de Mundos! Ejecuta el pipeline completo
#              con el Arsenal definitivo para la prueba de rendimiento final.
# -----------------------------------------------------------------
import pandas as pd
import numpy as np
import sys
import os
import time

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
# ¡AQUÍ ES DONDE LE DICES QUÉ YACIMIENTO ATACAR!
# Simplemente cambia el nombre del archivo .csv aquí antes de ejecutar.
DATASET_CSV_NAME = "marvin.csv"
# DATASET_CSV_NAME = "mclaughlin_limit.csv"
# DATASET_CSV_NAME = "SONDAJESCCC.csv"
# -----------------------------------------------------------------

def run_final_pipeline(input_csv_filename):
    """
    Función principal que encapsula y ejecuta todo el pipeline de modelado.
    """
    start_time = time.time()
    base_name = input_csv_filename.replace('.csv', '')
    print(f"--- Iniciando Pipeline 'Galactus' para el dataset: '{base_name}' ---")
    
    # --- PASO 0: Importaciones y configuración de ruta ---
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from pykrige.ok3d import OrdinaryKriging3D
        from scipy.spatial import cKDTree
        from tqdm import tqdm
        
        ruta_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if ruta_proyecto not in sys.path:
            sys.path.append(ruta_proyecto)
        # Importamos desde nuestro módulo de herramientas "Galactus"
        from src.evaluador_galactus import (
            calcular_features_para_punto, 
            calcular_features_ritmicas, 
            calcular_features_multifractales, 
            calcular_features_wavelet,
            trazar_veta_caotica,
            distancia_a_trayectoria,
            calcular_feature_potencial_ley
        )
        print("✅ Todas las librerías y módulos 'Galactus' cargados correctamente.")
    except ImportError as e:
        print(f"\n¡ERROR CRÍTICO! Fallo en la importación. Error: {e}")
        return

    # --- FASE 1: Carga y División de Datos ---
    print("\nFase 1: Cargando y dividiendo datos...")
    try:
        df_full = pd.read_csv(input_csv_filename)
        df_full.columns = df_full.columns.str.strip()
        
        # Guardamos una copia del dataset completo para la validación final
        df_full_original_predict = df_full.copy()
        
        df_train, df_predict = train_test_split(df_full, test_size=0.99, random_state=42)
        print(f"Datos de '{base_name}' cargados: {len(df_train)} para entrenamiento, {len(df_predict)} para validación.")
    except FileNotFoundError:
        print(f"¡ERROR! No se encontró el archivo '{input_csv_filename}'.")
        return

    # --- FASE 2: Calcular Predicciones de los Expertos (Nivel 0) ---
    print("\nFase 2: Generando predicciones de los modelos expertos...")
    X_sondajes_krig = df_train[['x', 'y', 'z']].values.astype(np.float64)
    y_sondajes_krig = df_train['au'].values.astype(np.float64)
    X_a_predecir_krig = df_predict[['x', 'y', 'z']].values.astype(np.float64)
    
    print("  - Calculando predicciones de Kriging...")
    ok3d_model = OrdinaryKriging3D(X_sondajes_krig[:,0], X_sondajes_krig[:,1], X_sondajes_krig[:,2], y_sondajes_krig, variogram_model='spherical', verbose=False)
    y_pred_kriging, _ = ok3d_model.execute('points', X_a_predecir_krig[:,0], X_a_predecir_krig[:,1], X_a_predecir_krig[:,2])
    RSE_KRIGING_BASELINE = np.sqrt(mean_squared_error(df_predict['au'], y_pred_kriging))
    print(f"✅ Predicciones de Kriging listas. RSE Baseline: {RSE_KRIGING_BASELINE:.5f}")

    # --- FASE 3: Generando Características del "Arsenal Galactus" ---
    print("  - Generando Características 'Galactus' (esto será épico)...")
    K_VECINOS_PARA_FEATURES = 64
    coords_sondajes = df_train[['x', 'y', 'z']].values
    leyes_sondajes = df_train['au'].values
    kdtree_global_sondajes = cKDTree(coords_sondajes)
    
    print("    -> Trazando vetas y calculando centroides (alta y baja ley)...")
    umbral_alta_ley = np.quantile(leyes_sondajes, 0.90)
    df_alta_ley = df_train[df_train['au'] > umbral_alta_ley]
    centroide_alta_ley = df_alta_ley[['x', 'y', 'z']].mean().values
    trayectoria_alta_ley = trazar_veta_caotica(df_alta_ley)
    
    umbral_baja_ley = np.quantile(leyes_sondajes, 0.10)
    df_baja_ley = df_train[df_train['au'] < umbral_baja_ley]
    centroide_baja_ley = df_baja_ley[['x', 'y', 'z']].mean().values
    trayectoria_baja_ley = trazar_veta_caotica(df_baja_ley)

    def generar_features_completas(puntos_a_procesar, desc):
        features_list = []
        for punto_coords in tqdm(puntos_a_procesar, desc=desc):
            _, indices = kdtree_global_sondajes.query(punto_coords, k=K_VECINOS_PARA_FEATURES, workers=-1)
            vecinos_coords, vecinos_leyes = coords_sondajes[indices], leyes_sondajes[indices]
            
            features_v12 = calcular_features_para_punto(punto_coords, K_VECINOS_PARA_FEATURES, kdtree_global_sondajes, coords_sondajes, leyes_sondajes)
            features_ritmicas = calcular_features_ritmicas(vecinos_coords, vecinos_leyes)
            features_multifractales = calcular_features_multifractales(vecinos_coords, vecinos_leyes)
            features_wavelet = calcular_features_wavelet(vecinos_coords, vecinos_leyes)
            features_galactus = {
                'distancia_centroide_alta': np.linalg.norm(punto_coords - centroide_alta_ley),
                'distancia_centroide_baja': np.linalg.norm(punto_coords - centroide_baja_ley),
                'distancia_veta_alta': distancia_a_trayectoria(punto_coords, trayectoria_alta_ley),
                'distancia_veta_baja': distancia_a_trayectoria(punto_coords, trayectoria_baja_ley),
                'potencial_por_m3': calcular_feature_potencial_ley(vecinos_coords, vecinos_leyes)
            }
            features_list.append({**features_v12, **features_ritmicas, **features_multifractales, **features_wavelet, **features_galactus})
        return pd.DataFrame(features_list)

    df_train_features = generar_features_completas(coords_sondajes, "Features 'Galactus' para Entrenamiento")
    df_train_enriquecido = pd.concat([df_train.reset_index(drop=True), df_train_features], axis=1)

    df_predict_features = generar_features_completas(df_predict[['x', 'y', 'z']].values, "Features 'Galactus' para Predicción")
    df_predict_enriquecido = pd.concat([df_predict.reset_index(drop=True), df_predict_features], axis=1)
    
    # Entrenar el modelo XGBoost "Galactus"
    FEATURES_GALACTUS = df_train_features.columns.tolist()
    print(f"\nEntrenando Experto XGBoost con {len(FEATURES_GALACTUS)} características del Arsenal 'Galactus'...")
    TARGET = 'au'
    X_train_galactus, y_train_galactus = df_train_enriquecido[FEATURES_GALACTUS], df_train_enriquecido[TARGET]
    X_predict_galactus = df_predict_enriquecido[FEATURES_GALACTUS]
    
    xgb_model_galactus = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, n_jobs=-1)
    xgb_model_galactus.fit(X_train_galactus, y_train_galactus, verbose=False)
    y_pred_xgb_galactus = xgb_model_galactus.predict(X_predict_galactus)
    print("✅ Predicciones de XGBoost 'Galactus' listas.")

    # --- FASE 4: Ensamblar el Mega-Dataset ---
    df_mega = df_predict_enriquecido.copy()
    df_mega['pred_kriging'] = y_pred_kriging
    df_mega['pred_xgb_galactus'] = y_pred_xgb_galactus
    print("✅ MEGA-DATASET para el 'Jefe Final' creado.")

    # --- FASE 5: Entrenar y Evaluar al "Jefe Final" ---
    print("\nFase 4: Entrenando y evaluando al 'Jefe Final'...")
    FEATURES_JEFE = FEATURES_GALACTUS + ['pred_kriging', 'pred_xgb_galactus']
    print(f"El 'Jefe Final' usará un total de {len(FEATURES_JEFE)} características de entrada.")
    X_jefe, y_jefe = df_mega[FEATURES_JEFE], df_mega[TARGET]
    X_train_jefe, X_test_jefe, y_train_jefe, y_test_jefe = train_test_split(X_jefe, y_jefe, test_size=0.2, random_state=42)
    
    xgb_jefe = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, early_stopping_rounds=50)
    xgb_jefe.fit(X_train_jefe, y_train_jefe, eval_set=[(X_test_jefe, y_test_jefe)], verbose=False)

    # --- VEREDICTO FINAL ---
    print("\nCalculando el RSE definitivo del 'Jefe Final'...")
    y_pred_final = xgb_jefe.predict(X_test_jefe)
    rse_final = np.sqrt(mean_squared_error(y_test_jefe, y_pred_final))

    print("\n========================================================")
    print(f"     🎉 VEREDICTO FINAL PARA '{base_name}' (MODELO GALACTUS) 🎉")
    print("========================================================")
    print(f"  RSE del 'Jefe Final' (Stacking con Arsenal Galactus): {rse_final:.5f}")
    print(f"  RSE de Kriging (Baseline)                           : {RSE_KRIGING_BASELINE:.5f}")
    print("========================================================")

    if rse_final < RSE_KRIGING_BASELINE:
        mejora = (1 - rse_final / RSE_KRIGING_BASELINE) * 100
        print(f"\n🚀🏆 ¡VICTORIA CÓSMICA! ¡El pipeline 'Galactus' superó al Kriging por un {mejora:.2f}%! 🏆🚀")
    else:
        print(f"\nEl Kriging ha defendido su título. ¡Incluso Galactus tiene sus límites!")
        
    end_time = time.time()
    print(f"\n--- Pipeline para '{base_name}' completado en {((end_time - start_time) / 60):.2f} minutos ---")

# --- El Guardián Mágico ---
if __name__ == "__main__":
    run_final_pipeline(DATASET_CSV_NAME)