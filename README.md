# __ML Modeling Challenge — Wizeline__

Solución al reto de regresión multivariada propuesto por Wizeline. El objetivo es predecir una variable continua a partir de 20 features, usando 800 muestras de entrenamiento y generando predicciones para 200 muestras del blind test dataset.

---

# __1. Configuración del entorno.__

## 1.1 Requisitos.

- Python 3.11
- [uv](https://github.com/astral-sh/uv) como gestor de entornos y paqueterías

## 1.2 Crear el ambiente virtual e instalar dependencias.

```bash
uv venv --python=python3.11
source .venv/bin/activate
uv sync
```

---

# __2. Sobre las tareas__


La lógica reutilizable está modularizada en tres scripts bajo `src/`:

- `src/eda_functions.py` — funciones de análisis exploratorio: distribuciones, correlaciones e Información Mutua.
- `src/train_functions.py` — construcción de pipelines, Cross-Validation, cálculo de métricas (SMAPE, MAPE, R², FUGACITY_SMAPE) y graficación de feature importances.
- `src/optimization_functions.py` — wrapper de `RandomizedSearchCV` con scorers personalizados.

A continuación se dará respuesta a las tareas propuestas.

#### __*2.1 Preprocesar las características si es necesario (justificar si no).*__

En __*1.Exploratory_Data_Analsis.ipynb*__ (ver las distintas __NOTAS__ a través del notebook) se dan las justificaciones, apoyadas en gráficos y métricas, acerca de por qué no es necesario hacer procedimientos de preprocesamiento explícito. Esto se debe a la ausencia de valores perdidos y de outliers, y a que las distribuciones de las features no presentan patrones problemáticos.

El único escalamiento aplicado es un `MinMaxScaler` embebido dentro de cada pipeline de entrenamiento, lo que garantiza que todas las features queden en la misma escala. Para las relaciones no lineales detectadas en el EDA, se recurre a modelos basados en árboles y boosting en lugar de transformaciones explícitas.

#### __*2.2 Seleccionar un subconjunto de características (justificar si no).*__

La selección de variables se realizó en dos etapas:

1. **Pre-selección vía Información Mutua** (en __*1.Exploratory_Data_Analsis.ipynb*__): se filtraron las features con mayor dependencia, lineal y no lineal, respecto al target, reduciendo las 20 features originales a un subconjunto inicial.

2. **Selección definitiva vía feature importances** (en __*2.Model_Competition_and_Variable_Selection.ipynb*__): se entrenó el modelo ganador (__CatBoost__) con las features pre-seleccionadas y se aplicó un umbral de importancia > 5, obteniendo 5 variables finales: `feature_2`, `feature_13`, `feature_9`, `feature_18`, `feature_11`. Pasar de las features de Información Mutua a estas 5 mejoró todas las métricas de Cross-Validation, validando la selección.

Es recomendable revisar las distintas __NOTAS__ a través de los notebooks.

#### __*2.3 Entrenar un modelo usando los datos de entrenamiento.*__

En __*2.Model_Competition_and_Variable_Selection.ipynb*__ se llevó a cabo una competencia entre 8 modelos: `Lasso`, `DecisionTree`, `RandomForest`, `ExtraTrees`, `GradientBoosting`, `XGBoost`, `LightGBM` y `CatBoost`. Todos fueron evaluados con k=5 Cross-Validation usando sus hiperparámetros por defecto. El ganador fue **CatBoost**, al obtener el menor SMAPE_CV, además del mayor R²_CV entre los competidores.

En __*3.Model_Optimization_and_Model_Saving.ipynb*__ se desarrolla un `RandomizedSearchCV` con 200 iteraciones para optimizar los hiperparámetros del __CatBoost__ (`iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `subsample`). El binario del modelo optimizado se guarda en `models/catboost_optimized.joblib` y las métricas en `config.yaml`.

Es recomendable revisar las distintas __NOTAS__ a través de los notebooks.

#### __*2.4 Reportar las métricas del modelo.*__

Se reportan dos métricas principales:
- **SMAPE** (métrica de selección): penaliza errores relativos de forma simétrica, robusta ante targets cercanos a cero.
- **R²**: mide la proporción de varianza explicada por el modelo.

El modelo __CatBoost__ optimizado tiene las siguientes métricas de Train y de Cross-Validation:

<div align="center">

| SMAPE_TRAIN | SMAPE_CV | R²_TRAIN | R²_CV |
|:-----------:|:--------:|:--------:|:-----:|
|    6.76 %   |  11.07 % |   0.96   | 0.89  |

</div>

Las métricas de CV son la referencia confiable del rendimiento generalizable del modelo. La brecha entre Train y CV es moderada, lo que indica que el modelo no presenta sobreajuste severo. Cabe destacar que el R²_CV de 0.89 es consistente con el máximo teórico esperable (0.92).

#### __*2.5 Predecir los valores objetivo para el blind test dataset.*__

Las predicciones se generan en __*4.Blind_Test_Data_Prediction.ipynb*__ cargando el modelo desde `models/catboost_optimized.joblib`. Como validación adicional, se comparó la distribución de las predicciones con la del target de entrenamiento mediante un histograma superpuesto, confirmando que ambas distribuciones tienen formas similares y el mismo rango (0-30), sin predicciones fuera de dominio.

Las predicciones para las 200 observaciones están localizadas en `data/predictions.csv` y la columna de valores predichos es `target_pred`.

---

# __3. Nota sobre estructura de producción.__

Este proyecto tiene una estructura orientada a la exploración y presentación del challenge (notebooks secuenciales), lo cual no es ideal para un entorno de producción.

Para consultar un ejemplo de cómo se vería un proyecto de Data Science con arquitectura end-to-end lista para producción, incluyendo versionamiento de modelos con **MLflow** y despliegue con **Docker + FastAPI**, puede consultarse el siguiente repositorio:

[github.com/miguel-uicab/good_bad_applicant_v2](https://github.com/miguel-uicab/good_bad_applicant_v2)


__¡Muchas Gracias!__