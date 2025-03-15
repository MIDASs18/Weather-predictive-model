import pandas as pd
import numpy as np
import joblib
import os
import datetime
import argparse
from dateutil.parser import parse

def cargar_modelo():
    """Carga el modelo entrenado, el escalador y las características"""
    try:
        modelo = joblib.load("model/rain_prediction_model.pkl")
        escalador = joblib.load("model/scaler.pkl")
        caracteristicas = joblib.load("model/features.pkl")
        return modelo, escalador, caracteristicas
    except FileNotFoundError:
        print("Error: No se encontraron los archivos del modelo.")
        print("Asegúrate de que los archivos 'rain_prediction_model.pkl', 'scaler.pkl' y 'features.pkl' estén en la carpeta model del directorio actual.")
        return None, None, None

def cargar_datos_historicos(ruta_archivo):
    """Carga los datos históricos para usarlos como referencia"""
    try:
        df = pd.read_csv(ruta_archivo)
        df['date'] = pd.to_datetime(df['date'])
        
        nulos = df.isnull().sum()
        if nulos.sum() > 0:
            print(f"Advertencia: El archivo contiene {nulos.sum()} valores nulos")
            print(nulos[nulos > 0])
            
            # Limpiar valores nulos en columnas críticas
            for col in ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'wdir']:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # Verificar si quedan valores nulos en columnas importantes
            nulos_restantes = df[['tavg', 'prcp', 'pres',]].isnull().sum().sum()
            if nulos_restantes > 0:
                print(f"Advertencia: Quedan {nulos_restantes} valores nulos en columnas críticas")
                print("Se eliminarán las filas con valores nulos en estas columnas")
                df = df.dropna(subset=['tavg', 'prcp', 'pres',])
            
            print(f"Datos listos para usar. Filas restantes: {len(df)}")
        
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en {ruta_archivo}")
        return None
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        return None

def preparar_datos_fecha(df_historico, fecha):
    """Prepara los datos para una fecha específica usando información similar de datos históricos"""
    if isinstance(fecha, str):
        fecha = pd.to_datetime(fecha)
    
    # Encontramos los datos históricos para el mismo mes y día (ignorando el año)
    datos_similares = df_historico[
        (df_historico['date'].dt.month == fecha.month) &
        (df_historico['date'].dt.day.isin([fecha.day-1, fecha.day, fecha.day+1]))
    ]
    
    if len(datos_similares) == 0:
        # Si no hay datos similares, usamos el mismo mes
        datos_similares = df_historico[df_historico['date'].dt.month == fecha.month]
    
    if len(datos_similares) == 0:
        print(f"No hay datos históricos similares para {fecha.strftime('%Y-%m-%d')}")
        return None
    
    # Verificamos valores nulos en datos_similares
    if datos_similares.isnull().sum().sum() > 0:
        for col in datos_similares.columns:
            if datos_similares[col].isnull().sum() > 0 and col != 'date':
                datos_similares[col] = datos_similares[col].fillna(datos_similares[col].mean())
    
    # Creamos un DataFrame para la fecha de predicción basado en promedios históricos
    datos_fecha = datos_similares.mean().to_dict()
    
    df_prediccion = pd.DataFrame([datos_fecha])
    
    df_prediccion['month'] = fecha.month
    df_prediccion['day_of_year'] = fecha.timetuple().tm_yday
    df_prediccion['day_of_week'] = fecha.weekday()
    df_prediccion['season'] = (fecha.month % 12 + 3) // 3
    
    # Calculamos las características derivadas que necesita el modelo
    # tavg_3d_mean (promedio de temperatura de 3 días)
    df_prediccion['tavg_3d_mean'] = df_prediccion['tavg']
    
    # prcp_3d_sum (suma de precipitación de 3 días)
    df_prediccion['prcp_3d_sum'] = df_prediccion['prcp']
    
    # pres_diff (diferencia de presión)
    df_prediccion['pres_diff'] = 0
    
    # Calculamos componentes trigonométricas de la dirección del viento
    if 'wdir' in df_prediccion.columns:
        df_prediccion['wdir_sin'] = np.sin(np.radians(df_prediccion['wdir']))
        df_prediccion['wdir_cos'] = np.cos(np.radians(df_prediccion['wdir']))
    else:
        df_prediccion['wdir_sin'] = 0
        df_prediccion['wdir_cos'] = 0
    
    return df_prediccion

def predecir_probabilidad_lluvia(modelo, escalador, caracteristicas, df_datos):
    """Predice la probabilidad de lluvia usando el modelo"""
    try:
        # Verificamos todas las características necesarias
        missing_features = [feat for feat in caracteristicas if feat not in df_datos.columns]
        if missing_features:
            print(f"Error: Faltan las siguientes características: {missing_features}")
            return None, None
            
        X = df_datos[caracteristicas]
        
        X_scaled = escalador.transform(X)
        
        # Predecimos la probabilidad
        probabilidad = modelo.predict_proba(X_scaled)[:, 1][0]
        prediccion = modelo.predict(X_scaled)[0]
        
        return prediccion, probabilidad * 100
    except Exception as e:
        print(f"Error al realizar la predicción: {str(e)}")
        return None, None

def interpretar_probabilidad(probabilidad):
    """Interpreta la probabilidad de lluvia en texto"""
    if probabilidad is None:
        return "No se pudo determinar"
    elif probabilidad < 20:
        return "Muy baja"
    elif probabilidad < 40:
        return "Baja"
    elif probabilidad < 60:
        return "Moderada"
    elif probabilidad < 80:
        return "Alta"
    else:
        return "Muy alta"

def generar_pronostico_rango(df_historico, modelo, escalador, caracteristicas, fecha_inicio, fecha_fin):
    """Genera pronóstico para un rango de fechas"""
    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
    resultados = []
    
    for fecha in fechas:
        df_fecha = preparar_datos_fecha(df_historico, fecha)
        if df_fecha is not None:
            prediccion, probabilidad = predecir_probabilidad_lluvia(modelo, escalador, caracteristicas, df_fecha)
            resultados.append({
                'fecha': fecha.strftime('%Y-%m-%d'),
                'prediccion': 'Lluvia' if prediccion == 1 else 'No lluvia',
                'probabilidad': probabilidad,
                'interpretacion': interpretar_probabilidad(probabilidad)
            })
    
    return resultados

def analisis_mensual(df_historico, modelo, escalador, caracteristicas, año, mes):
    """Genera análisis de probabilidad de lluvia para un mes específico"""
    ultimo_dia = pd.Timestamp(year=año, month=mes, day=1) + pd.offsets.MonthEnd(1)
    dias_en_mes = ultimo_dia.day
    
    fecha_inicio = pd.Timestamp(year=año, month=mes, day=1)
    fecha_fin = pd.Timestamp(year=año, month=mes, day=dias_en_mes)
    
    return generar_pronostico_rango(df_historico, modelo, escalador, caracteristicas, fecha_inicio, fecha_fin)

def interfaz_linea_comandos():
    """Interfaz de línea de comandos para la aplicación"""
    parser = argparse.ArgumentParser(description='Predictor de lluvia')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fecha', help='Fecha específica (YYYY-MM-DD)')
    group.add_argument('--rango', nargs=2, metavar=('INICIO', 'FIN'), help='Rango de fechas (YYYY-MM-DD YYYY-MM-DD)')
    group.add_argument('--mes', nargs=2, type=int, metavar=('AÑO', 'MES'), help='Año y mes para análisis (YYYY MM)')
    
    parser.add_argument('--datos', default='datos_procesados.csv', help='Ruta al archivo CSV con datos históricos')
    
    args = parser.parse_args()
    
    modelo, escalador, caracteristicas = cargar_modelo()
    if modelo is None:
        return
    
    df_historico = cargar_datos_historicos(args.datos)
    if df_historico is None:
        return
    
    if args.fecha:
        try:
            fecha = parse(args.fecha)
            df_fecha = preparar_datos_fecha(df_historico, fecha)
            if df_fecha is not None:
                prediccion, probabilidad = predecir_probabilidad_lluvia(modelo, escalador, caracteristicas, df_fecha)
                print(f"\nPredicción para {fecha.strftime('%Y-%m-%d')}:")
                print(f"Resultado: {'Lluvia' if prediccion == 1 else 'No lluvia'}")
                print(f"Probabilidad de lluvia: {probabilidad:.2f}%")
                print(f"Interpretación: {interpretar_probabilidad(probabilidad)}")
        except ValueError:
            print(f"Error: Formato de fecha incorrecto. Use YYYY-MM-DD.")
    elif args.rango:
        try:
            fecha_inicio = parse(args.rango[0])
            fecha_fin = parse(args.rango[1])
            
            if fecha_inicio > fecha_fin:
                print("Error: La fecha de inicio debe ser anterior a la fecha de fin.")
                return
                
            resultados = generar_pronostico_rango(df_historico, modelo, escalador, caracteristicas, fecha_inicio, fecha_fin)
            
            print(f"\nPronóstico de lluvia del {fecha_inicio.strftime('%Y-%m-%d')} al {fecha_fin.strftime('%Y-%m-%d')}:")
            print("-" * 80)
            print(f"{'Fecha':<12} | {'Predicción':<10} | {'Probabilidad':<15} | {'Interpretación':<15}")
            print("-" * 80)
            
            for resultado in resultados:
                print(f"{resultado['fecha']:<12} | {resultado['prediccion']:<10} | {resultado['probabilidad']:.2f}%{' ':<10} | {resultado['interpretacion']:<15}")
            
        except ValueError:
            print(f"Error: Formato de fecha incorrecto. Use YYYY-MM-DD.")
    elif args.mes:
        año, mes = args.mes
        if mes < 1 or mes > 12:
            print("Error: El mes debe estar entre 1 y 12.")
            return
            
        resultados = analisis_mensual(df_historico, modelo, escalador, caracteristicas, año, mes)
        
        nombre_mes = pd.Timestamp(year=año, month=mes, day=1).strftime('%B')
        print(f"\nAnálisis de lluvia para {nombre_mes} de {año}:")
        print("-" * 80)
        print(f"{'Fecha':<12} | {'Predicción':<10} | {'Probabilidad':<15} | {'Interpretación':<15}")
        print("-" * 80)
        
        for resultado in resultados:
            print(f"{resultado['fecha']:<12} | {resultado['prediccion']:<10} | {resultado['probabilidad']:.2f}%{' ':<10} | {resultado['interpretacion']:<15}")
        
        probabilidades = [r['probabilidad'] for r in resultados]
        dias_lluvia = sum(1 for r in resultados if r['prediccion'] == 'Lluvia')
        
        print("-" * 80)
        print(f"Resumen para {nombre_mes} {año}:")
        print(f"Días con pronóstico de lluvia: {dias_lluvia} de {len(resultados)} ({dias_lluvia/len(resultados)*100:.1f}%)")
        print(f"Probabilidad media de lluvia: {sum(probabilidades)/len(probabilidades):.2f}%")
        print(f"Probabilidad máxima: {max(probabilidades):.2f}%")
        print(f"Probabilidad mínima: {min(probabilidades):.2f}%")

def interfaz_interactiva():
    """Interfaz interactiva para la aplicación"""
    print("\n=== PREDICTOR DE LLUVIA ===")
    
    print("\nCargando modelo y datos históricos...")
    modelo, escalador, caracteristicas = cargar_modelo()
    if modelo is None:
        return
    
    ruta_datos = input("\nRuta al archivo CSV con datos históricos (Enter para usar 'datos_procesados.csv'): ")
    if not ruta_datos:
        ruta_datos = 'datos_procesados.csv'
    
    df_historico = cargar_datos_historicos(ruta_datos)
    if df_historico is None:
        return
    
    print("\nModelo y datos cargados correctamente.")
    
    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Predicción para una fecha específica")
        print("2. Predicción para un rango de fechas")
        print("3. Análisis mensual")
        print("4. Salir")
        
        opcion = input("\nSeleccione una opción (1-4): ")
        
        if opcion == '1':
            fecha_str = input("\nIngrese la fecha (YYYY-MM-DD): ")
            try:
                fecha = parse(fecha_str)
                df_fecha = preparar_datos_fecha(df_historico, fecha)
                if df_fecha is not None:
                    prediccion, probabilidad = predecir_probabilidad_lluvia(modelo, escalador, caracteristicas, df_fecha)
                    print(f"\nPredicción para {fecha.strftime('%Y-%m-%d')}:")
                    print(f"Resultado: {'Lluvia' if prediccion == 1 else 'No lluvia'}")
                    if probabilidad is not None:
                        print(f"Probabilidad de lluvia: {probabilidad:.2f}%")
                    else:
                        print("Probabilidad de lluvia: No disponible")
                    print(f"Interpretación: {interpretar_probabilidad(probabilidad)}")
            except ValueError:
                print("Error: Formato de fecha incorrecto. Use YYYY-MM-DD.")
        elif opcion == '2':
            try:
                fecha_inicio_str = input("\nIngrese la fecha de inicio (YYYY-MM-DD): ")
                fecha_fin_str = input("Ingrese la fecha de fin (YYYY-MM-DD): ")
                
                fecha_inicio = parse(fecha_inicio_str)
                fecha_fin = parse(fecha_fin_str)
                
                if fecha_inicio > fecha_fin:
                    print("Error: La fecha de inicio debe ser anterior a la fecha de fin.")
                    continue
                    
                resultados = generar_pronostico_rango(df_historico, modelo, escalador, caracteristicas, fecha_inicio, fecha_fin)
                
                print(f"\nPronóstico de lluvia del {fecha_inicio.strftime('%Y-%m-%d')} al {fecha_fin.strftime('%Y-%m-%d')}:")
                print("-" * 80)
                print(f"{'Fecha':<12} | {'Predicción':<10} | {'Probabilidad':<15} | {'Interpretación':<15}")
                print("-" * 80)
                
                for resultado in resultados:
                    print(f"{resultado['fecha']:<12} | {resultado['prediccion']:<10} | {resultado['probabilidad']:.2f}%{' ':<10} | {resultado['interpretacion']:<15}")
                
            except ValueError:
                print("Error: Formato de fecha incorrecto. Use YYYY-MM-DD.")
        elif opcion == '3':
            try:
                año = int(input("\nIngrese el año: "))
                mes = int(input("Ingrese el mes (1-12): "))
                
                if mes < 1 or mes > 12:
                    print("Error: El mes debe estar entre 1 y 12.")
                    continue
                    
                resultados = analisis_mensual(df_historico, modelo, escalador, caracteristicas, año, mes)
                
                nombre_mes = pd.Timestamp(year=año, month=mes, day=1).strftime('%B')
                print(f"\nAnálisis de lluvia para {nombre_mes} de {año}:")
                print("-" * 80)
                print(f"{'Fecha':<12} | {'Predicción':<10} | {'Probabilidad':<15} | {'Interpretación':<15}")
                print("-" * 80)
                
                for resultado in resultados:
                    print(f"{resultado['fecha']:<12} | {resultado['prediccion']:<10} | {resultado['probabilidad']:.2f}%{' ':<10} | {resultado['interpretacion']:<15}")
                
                # Calcular estadísticas
                probabilidades = [r['probabilidad'] for r in resultados]
                dias_lluvia = sum(1 for r in resultados if r['prediccion'] == 'Lluvia')
                
                print("-" * 80)
                print(f"Resumen para {nombre_mes} {año}:")
                print(f"Días con pronóstico de lluvia: {dias_lluvia} de {len(resultados)} ({dias_lluvia/len(resultados)*100:.1f}%)")
                print(f"Probabilidad media de lluvia: {sum(probabilidades)/len(probabilidades):.2f}%")
                print(f"Probabilidad máxima: {max(probabilidades):.2f}%")
                print(f"Probabilidad mínima: {min(probabilidades):.2f}%")
                
            except ValueError:
                print("Error: Por favor ingrese valores numéricos válidos.")
        elif opcion == '4':
            print("\n¡Gracias por usar el Predictor de Lluvia! - Francisco Bracamonte Bravo")
            break
        else:
            print("\nOpción no válida. Por favor seleccione una opción del 1 al 4.")

if __name__ == "__main__":
    # Intentamos usar la interfaz de línea de comandos si hay argumentos
    import sys
    if len(sys.argv) > 1:
        interfaz_linea_comandos()
    else:
        interfaz_interactiva()
