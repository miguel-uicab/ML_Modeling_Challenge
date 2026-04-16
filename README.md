# __ML_Modeling_Challenge__



# 1.Sobre las tareas.

## __1.1 Preprocesar las características si es necesario (justificar si no).__
En __*1.Exploratory_Data_Analsis.ipynb*__ (ver las distintas__NOTAS__ a través del notebook) se dan las justificaciones, apoyadas en gráficos y métricas, acerca de porqué no es necesario hacer procedimientos de preprocesamiento. Esto es debido, básicamente, a la no presencia de valores perdidos ni outliers.
Además, para hacer frente a las relaciones no lineales localizadas, se recurre al uso de modelos basados en árboles o boosting en la parte de competencia de modelos.

## __1.2 Seleccionar un subconjunto de características (justificar si no).__
En __*1.Exploratory_Data_Analsis.ipynb*__ se hace una pre-selección de variables usando la Información Mutua, la cual es capaz de detectar dependencias (relaciones) tanto lineales como no lineales con respecto al target.

Sin embargo, una selección de variables definitiva se alcanza en el notebook __*2.Model_Competition_and_Varible_Selection.ipynb*__ resultante de la obtención de las feature importances del modelo ganador, es decir, un CatBoost. La variables seleccionadas, a saber, son: `feature 2`, `feature 13`, `feature 9`, `feature_18`, `feature_11`.

## __1.3 Entrenar un modelo usando los datos de entrenamiento.__
En __*2.Model_Competition_and_Varible_Selection.ipynb*__ lleva a cabo una competencia de modelos. El ganador ha sido un Catboost al alcanzar valores de SMAPE de cross-validation más chicos que sus competidores (ver las distintas__NOTAS__ a través del notebook).

En __*3.Model_Optimization_and_Model_Saving.ipynb*__ se desarrolla un `RandomSearchCV` para optimziar el Catboost. Se guardan las métricas y el binario del modelo optimizado.

## __1.4 Reportar las métricas del modelo que consideres necesarias o adecuadas para evaluar su rendimiento. El objetivo tiene algo de ruido, aunque si encontraras la función exacta sin ruido obtendrías alrededor de 0.92 R².__
El modelo Catboost optimizado tiene las siguientes métricas de Cross-Validation:

| SMAPE_CV | R2_CV  |
|----------|--------|
| 11.2 %   | 0.8862 |

## __1.5 Predecir los valores objetivo para el blind test dataset.__

Las predicciones para las 200 observaciones en el blind test dataset están localizadas en `data/predictions`.csv  y la columna de valores predichos es `target_pred`.