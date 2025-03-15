# Predicción de Lluvia con RandomForest

Este proyecto utiliza un modelo de Machine Learning basado en **RandomForest** para predecir la probabilidad de lluvia en una fecha determinada, utilizando datos históricos del clima. La aplicación permite realizar predicciones individuales, analizar un rango de fechas y obtener un análisis mensual de probabilidades de lluvia.

## 1. Limpieza de Datos

El conjunto de datos original contenía información meteorológica como temperatura media, mínima y máxima, precipitación, velocidad del viento, presión atmosférica y dirección del viento. Antes de entrenar el modelo, se realizaron los siguientes pasos de limpieza:

- **Conversión de fechas**: Se transformaron las fechas a un formato estándar (`datetime`).
- **Detección y tratamiento de valores nulos**:
  - Las columnas críticas (`tavg`, `tmin`, `tmax`, `prcp`, `wspd`, `pres`, `wdir`) se interpolaron usando un método lineal, con relleno hacia atrás y hacia adelante.
  - Si después de la interpolación aún quedaban valores nulos en `tavg`, `prcp` o `pres`, se eliminaron las filas correspondientes.
- **Generación de nuevas características**:
  - **Promedio móvil de temperatura (tavg_3d_mean)**: Promedio de los últimos tres días.
  - **Suma acumulada de precipitación (prcp_3d_sum)**: Suma de los últimos tres días.
  - **Diferencia de presión atmosférica (pres_diff)**: Diferencia con respecto al día anterior.
  - **Descomposición trigonométrica de la dirección del viento** (`wdir_sin`, `wdir_cos`).

Tras estos pasos, los datos estaban listos para el análisis y entrenamiento del modelo.

## 2. Análisis de Datos

Antes de construir el modelo, se realizó un análisis exploratorio para comprender mejor las relaciones entre las variables:

- **Correlaciones**: Se analizó la correlación entre la precipitación (`prcp`) y otras variables, observando que la temperatura y la presión atmosférica tenían una fuerte relación con la presencia de lluvia.
- **Distribuciones**: Se examinaron las distribuciones de temperatura y precipitación para detectar valores atípicos y tendencias estacionales.
- **Estacionalidad**: Se identificaron patrones estacionales en los datos, como mayor probabilidad de lluvia en ciertos meses del año.

Este análisis ayudó a definir las características más relevantes para el modelo.

## 3. Elección de RandomForest

Se optó por un **RandomForestClassifier** debido a las siguientes razones:

- **Manejo de datos faltantes y ruido**: RandomForest es robusto ante valores atípicos y datos faltantes parciales.
- **Capacidad de manejar datos no lineales**: Al ser un conjunto de árboles de decisión, captura relaciones no lineales entre las variables.
- **Importancia de características**: Permite evaluar qué variables son más relevantes para la predicción de lluvia.
- **Alto rendimiento sin necesidad de ajuste excesivo**: Funciona bien sin requerir una optimización extrema de hiperparámetros.

## 4. Entrenamiento del Modelo

El modelo se entrenó con los datos históricos procesados siguiendo estos pasos:

1. **División del dataset**: Se separaron los datos en entrenamiento (80%) y prueba (20%).
2. **Escalado de variables**: Se aplicó una normalización con `StandardScaler` para mejorar el rendimiento del modelo.
3. **Entrenamiento del RandomForestClassifier**: Se ajustó con una cantidad óptima de árboles (`n_estimators=200`) y una profundidad controlada (`max_depth=None`) para evitar sobreajuste.
4. **Evaluación del modelo**: Se obtuvo una precisión alta en la clasificación de lluvia, y se utilizó la métrica de **ROC-AUC** para evaluar el rendimiento.

El modelo resultante se guardó en `rain_prediction_model.pkl`, junto con el escalador (`scaler.pkl`) y las características utilizadas (`features.pkl`).

## 5. Funcionamiento de la Aplicación

La aplicación permite predecir la lluvia a partir de datos históricos y la fecha seleccionada. Funciona de la siguiente manera:

1. **Carga del modelo y datos**: Se cargan el modelo entrenado, el escalador y las características necesarias.
2. **Selección de fecha o rango de fechas**:
   - Predicción para un día específico.
        · El usuario ingresa una fecha en formato YYYY-MM-DD.
        · La aplicación busca datos históricos similares (mismo mes y día).
        · Muestra la predicción (lluvia/no lluvia) y la probabilidad.
   - Análisis de un rango de fechas.
        · El usuario ingresa una fecha de inicio y una fecha de fin.
        · La aplicación genera predicciones para cada día en ese rango.
        · Muestra una tabla con los resultados.
   - Análisis mensual.
        · El usuario ingresa un año y un mes.
        · La aplicación genera predicciones para todos los días de ese mes.
        · Muestra un resumen con estadísticas como:
             - Días con pronóstico de lluvia.
             - Probabilidad media, máxima y mínima.
3. **Preparación de los datos**:
   - Se extraen datos históricos de fechas similares.
   - Se generan las mismas características utilizadas en el entrenamiento.
   - Se escalan las variables antes de pasarlas al modelo.
4. **Predicción de probabilidad de lluvia**:
   - Se obtiene la predicción (`Lluvia` o `No lluvia`).
   - Se calcula la probabilidad de lluvia en porcentaje.
   - Se genera una interpretación de la probabilidad (`Muy baja`, `Baja`, `Moderada`, etc.).

El usuario puede interactuar con la aplicación desde la línea de comandos proporcionando una fecha o un rango de fechas. También se incluye una opción interactiva para facilitar el uso.

