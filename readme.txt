# Dia 28 GEa AI: Estimación de Recursos Híbrida de Próxima Generación

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**GEa AI** es un proyecto revolucionario en Python que redefine el paradigma de la estimación de recursos minerales. En lugar de elegir entre la geoestadística tradicional y el Machine Learning, hemos construido un **pipeline híbrido de "stacking"** que fusiona lo mejor de ambos mundos para lograr una precisión sin precedentes.

Este proyecto nació del desafío de superar las limitaciones inherentes del Kriging Ordinario, como el efecto de suavizado y la dependencia de la estacionariedad. El resultado es un motor de estimación que ha demostrado mejoras de hasta un **51% en la reducción del Error Cuadrático Medio (RMSE)** en yacimientos complejos.

---

## 🚀 Características Principales

-   **Ingeniería de Características "Galactus":** Genera automáticamente un arsenal de más de 20 características geológicamente significativas para cada punto, incluyendo:
    -   Análisis de **Anisotropía y Gradientes 3D Locales**.
    -   Análisis de **Ritmicidad Espacial** mediante Transformada de Fourier (FFT).
    -   Análisis de **Complejidad Multifractal** (MFDFA).
    -   Análisis **Wavelet Multiescala** para capturar la "energía" del yacimiento.
    -   Modelado de **"Vetas Caóticas"** y centroides de alta/baja ley.

-   **Arquitectura de Stacking ("Jefe Final"):** Utiliza un meta-modelo XGBoost que aprende a combinar de forma inteligente las predicciones de dos expertos:
    1.  **Experto Geoestadístico:** Kriging Ordinario 3D.
    2.  **Experto en Patrones:** Un modelo XGBoost entrenado con el "Arsenal Galactus".

-   **Pipeline Generalizable:** Diseñado para ser una "fábrica de algoritmos". El mismo pipeline se ha validado en múltiples y diversos datasets, demostrando su robustez y superioridad.

-   **Optimización por RL (I+D Interno):** Incluye un framework para usar Aprendizaje por Refuerzo (`Stable-Baselines3`) para la optimización automatizada de los hiperparámetros de todo el pipeline.

---

## 🛠️ Instalación y Uso

Sigue estos pasos para ejecutar el pipeline de GEa AI en tu máquina local.

### Prerrequisitos

-   Python 3.8 o superior.
-   Git instalado en tu sistema.
-   (Opcional, para aceleración) Una GPU NVIDIA con CUDA Toolkit instalado.

### Pasos

1.  **Clona este repositorio:**
    ```bash
    git clone https://github.com/TheBullMiner/28DiasDePythonParaMineria/tree/main]
    cd [28DiasDePythonParaMineria]
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    python -m venv .venv
    # En Windows:
    .\.venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    Las librerías necesarias para GEa AI.
    ```bash
	numpy==1.26.4
	pandas==2.2.2
	scikit-learn==1.4.2
	xgboost==2.0.3
	pykrige==1.7.2
	scipy==1.13.0
	tqdm==4.66.4
	MFDFA==0.4.1
	PyWavelets==1.6.0
	matplotlib==3.8.4
	stable-baselines3==2.3.0
	gymnasium==0.29.1
	tensorflow==2.16.1
    ```

4.  **Ejecuta el pipeline principal:**
    El script `31_jefe_final_galactus.py` es la herramienta principal. Abre el script y configura la variable `DATASET_CSV_NAME` para apuntar a tu archivo de sondajes.
    ```bash
    python scripts/31_jefe_final_galactus.py
    ```
    La salida será un veredicto en la consola comparando el RSE de GEa AI con el del Kriging.

---

## 🔧 Cómo Adaptarlo a Tus Datos

Nuestra "Fábrica de Algoritmos" está diseñada para ser universal.
1.  **Prepara tus Datos:** Asegúrate de que tu CSV de sondajes contenga, como mínimo, las columnas `x`, `y`, `z`, y `au` (o el nombre de la ley que quieras predecir).
2.  **Configura el Script:** Abre el `script 31` y cambia el valor de `DATASET_CSV_NAME` al nombre de tu archivo.
3.  **¡Ejecuta!** El pipeline se encargará del resto, desde la división de datos hasta el veredicto final.

---

## 💼 Servicios Profesionales y Colaboración (GEa AI)

Si bien los scripts de validación de este proyecto están liberados bajo una licencia de código abierto, la metodología completa y el pipeline optimizado son el núcleo de nuestra startup, **GEa AI**.

Ofrecemos servicios de consultoría para implementar esta tecnología de vanguardia y generar un valor tangible en su operación minera.

-   **Análisis de Potencial de Optimización:** Evaluamos sus datos actuales y le entregamos un informe que cuantifica la mejora porcentual que nuestra metodología puede alcanzar para su yacimiento específico.
-   **Generación de Modelos de Bloques de Alta Precisión:** Aplicamos nuestro pipeline completo para generar un modelo de recursos con su ley estimada y nuestra métrica de incertidumbre patentada.
-   **Implementación de "Modelos Vivos":** Desarrollamos sistemas que se re-entrenan y actualizan automáticamente con los nuevos datos de producción (pozos de voladura, etc.), proporcionando una herramienta de soporte a la decisión para la planificación a corto plazo.

**¿Listo para reducir su incertidumbre y maximizar su rentabilidad? Hablemos.**

Contáctenos en **[benavidesmaycol81@gmail.com]** o a través del perfil de **Mayc Benavides (https://www.linkedin.com/in/maycolbenavidess/)**.

---

## 📜 Licencia

Los scripts de validación y los módulos de cálculo de este proyecto (`31_jefe_final_galactus.py` y `evaluador_galactus.py`) están liberados bajo la **Licencia Pública General de GNU v3.0 (GPLv3)**. Esto significa que eres libre de usar, estudiar, compartir y modificar el software para tus propios fines de investigación y validación.

Cualquier uso comercial, derivación para productos o implementación en entornos de producción requiere un acuerdo de licencia comercial con **GEa AI**.

Ver el archivo `LICENSE` para más detalles.