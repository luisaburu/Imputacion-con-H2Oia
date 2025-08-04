import glob
import tqdm
import pandas as pd  
from scipy.stats import linregress, pearsonr, spearmanr                 # Manejo de tablas y DataFrames
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import os
from scipy.stats import shapiro

def observaciones(nom_bd, nom_var, pathin, pathout):
    """
    Lee las series por estación para una variable dada,
    calcula estadísticas de completitud y exporta un resumen como CSV.
    """
    rutas = glob.glob(f"{pathin}/{nom_bd}/{nom_var}*")
    v = []

    for i in tqdm.tqdm(rutas):
        codigo = i.split('@')[1].split('.')[0]
        df = pd.read_csv(i, sep='|')

        if len(df) <= 100:
            continue

        df.columns = ['fecha', 'valor']
        df = df.drop(0).reset_index(drop=True)
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')

        # Estadísticas básicas
        fecha_ini = df['fecha'].min()
        fecha_fin = df['fecha'].max()
        total_datos = len(df)

        #print(f"\n📊 Estación {codigo} — Estadísticas de la serie:")
        #print(f"   - Fecha inicial       : {fecha_ini.strftime('%Y-%m-%d')}")
        #print(f"   - Fecha final         : {fecha_fin.strftime('%Y-%m-%d')}")
        #print(f"   - Total de datos      : {total_datos:,}")

        # Frecuencia temporal y datos esperados
        df['diferencia'] = df['fecha'].diff().dropna()
        frecuencia_promedio = df['diferencia'].mode()[0]
        minutos_abs = int(abs(frecuencia_promedio.total_seconds() / 60))

        fecha_inicial, fecha_final = df['fecha'].min(), df['fecha'].max()
        fechas_esperadas = pd.date_range(start=fecha_inicial, end=fecha_final, freq=f"{minutos_abs}min")
        conteo_esperado = len(fechas_esperadas)
        conteo_real = df['fecha'].nunique()
        datos_perdidos = conteo_esperado - conteo_real
        porcentaje_perdidos = (datos_perdidos / conteo_esperado) * 100

        if porcentaje_perdidos < 0:
            continue

        v.append([codigo, fecha_inicial, fecha_final, porcentaje_perdidos, total_datos, conteo_esperado])

    # Exportar resumen
    v = pd.DataFrame(v, columns=['cod', 'fi', 'ff', 'pdp', 'n', 'N'])
    v.to_csv(f'{pathout}/analisis_{nom_var}.csv', index=False)
    return v
def correlacion_hexbin(df, nom_var, path, nombre_carpeta):
    """
    Genera un gráfico hexbin que muestra la relación entre
    el tamaño de las series y el porcentaje de datos perdidos.
    Incluye regresión lineal y correlaciones Pearson y Spearman.
    """
    x = df['N']
    y = df['pdp']

    # Regresión lineal
    slope, intercept, r_value, _, _ = linregress(x, y)
    y_pred = slope * x + intercept

    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    # Gráfico
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(x, y, gridsize=30, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Número de estaciones')

    sorted_idx = x.argsort()
    plt.plot(x.iloc[sorted_idx], y_pred.iloc[sorted_idx], color='red', linewidth=2,
             label=f'Regresión lineal\n$R^2$ = {r_value**2:.2f}')

    plt.xlabel('Cantidad total de datos (N)')
    plt.ylabel('Porcentaje de datos perdidos (%)')
    plt.title(f'Relación entre tamaño de serie y % de datos perdidos\n{nom_var}')
    plt.grid(True, alpha=0.3)

    textstr = '\n'.join((
        f'Pearson r = {r_pearson:.2f} (p = {p_pearson:.4f})',
        f'Spearman ρ = {r_spearman:.2f} (p = {p_spearman:.4f})'
    ))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    path_out = os.path.join(path, nombre_carpeta, 'GRAFICOS', f'correlacion_hexbin_{nom_var}.png')
    plt.savefig(path_out, dpi=300)
    plt.show()
def seleccion(df_e, num, nom_bd, nom_var, pathin):
    """
    Filtra estaciones con al menos `num` datos y construye
    un DataFrame multiserie con los datos válidos.
    """
    v1 = df_e[df_e['n'] >= num].reset_index(drop=True)
    df_resultado = pd.DataFrame()
    count = 0

    rutas = glob.glob(f"{pathin}/{nom_bd}/{nom_var}*")

    for i in tqdm.tqdm(rutas):
        codigo = i.split('@')[1].split('.')[0]
        if codigo not in v1['cod'].values:
            continue

        df = pd.read_csv(i, sep='|')
        if len(df) <= 100:
            continue

        df.columns = ['fecha', 'valor']
        df = df.drop(0).reset_index(drop=True)
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        df = df[(df['fecha'] >= datetime.datetime(1960, 1, 1)) & (df['fecha'] <= datetime.datetime(2026, 1, 1))]

        if len(df) <= 100:
            continue

        df = df.set_index('fecha')
        df.columns = [f'{codigo}']

        print(f"📌 Estación añadida: {codigo} ({count})")
        if count == 0:
            df_resultado = df
        else:
            df_resultado = pd.concat([df_resultado, df], axis=1)

        count += 1

    return df_resultado
def Analisis(nom_bd, nom_var, path_in, path_out):
    """
    Ejecuta el análisis completo para una sola variable:
    - Carga y evalúa las series
    - Exporta estadísticas
    - Genera gráfico de correlación
    """
    df = observaciones(nom_bd, nom_var, path_in, path_out)
    correlacion_hexbin(df, nom_var, path_out, nom_bd)
    return df
def Analisis_Completo(path_in, path_out):
    """
    Ejecuta el análisis para todas las variables hidrológicas,
    meteorológicas y de temperatura predefinidas.
    """
    print(f" Iniciando análisis sobre datos en: {path_in}")

    var_s_hidro = ['Q_MEDIA_D', 'Q_MN_D', 'Q_MX_D']
    var_s_met = ['BSHG_TT_D', 'DVAG_CON', 'EVTE_CON', 'FA_CON', 'HR_CAL_MEDIA_D',
                 'NB_CON', 'RCAM_CON', 'TPR_CAL', 'TV_CAL', 'VVAG_CON']
    var_s_tem = ['TSSM_CON', 'TSSM_MEDIA_D', 'TSSM_MN_D', 'TSSM_MX_D']

    for i in var_s_hidro:
        Analisis('HidrologiaNacionalDiaria', i, path_in, path_out)

    # Descomentar si quieres incluir meteorología
    # for i in var_s_met:
    #     Analisis('MetereologiaNacionalDiaria', i, path_in, path_out)

    Analisis('PrecipitacionNacionalDiaria', 'PTPM_CON_INTER', path_in, path_out)

    for i in var_s_tem:
        Analisis('TemperaturaNacionalDiaria', i, path_in, path_out)
def ObtenerDatos(path, tipo):
    """
    Carga, limpia y transforma una serie de tiempo desde un archivo CSV de IDEAM.
    Realiza resampleo diario (por promedio o suma), y entrega estadísticas básicas.

    Parámetros:
    - path (str): ruta al archivo CSV de entrada (separado por '|')
    - tipo (str): 'promediada' o 'acumulada', define el tipo de agregación al resamplear

    Retorna:
    - df (DataFrame): serie de tiempo procesada con columnas ['fecha', 'valor']
    """

    # Cargar archivo CSV con separador '|'
    df = pd.read_csv(path, sep='|')
    
    # Asignar nombres a las columnas
    df.columns = ['fecha', 'valor']
    
    # Eliminar primera fila (que usualmente es redundante en archivos IDEAM)
    df = df.drop(0).reset_index(drop=True)
    
    # Conversión de columnas a formatos adecuados
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')  # a datetime
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')   # a número

    # Filtrar fechas a partir de 1969 (usualmente se descartan registros previos)
    df = df[df['fecha'] >= datetime.datetime(1969, 1, 1)]
    
    # Reindexar por fecha para aplicar resample
    df = df.set_index('fecha')

    # Agregación diaria:
    # - Si es una variable promediada (ej: temperatura), usar promedio
    # - Si es acumulada (ej: precipitación), usar suma
    if tipo == 'promediada':
        df = df.resample('D').mean().fillna(0).rename(columns={'valor': 'valor'})
    elif tipo == 'acumulada':
        df = df.resample('D').sum().fillna(0).rename(columns={'valor': 'valor'})
    
    # Restaurar el índice numérico
    df = df.reset_index()

    # Estadísticas básicas
    fecha_ini = df['fecha'].min()
    fecha_fin = df['fecha'].max()
    total_datos = len(df)

    # 📊 Mostrar estadísticas formateadas
    print("📊 Estadísticas generales de la serie")
    print(f"- Fecha inicial: {fecha_ini.strftime('%Y-%m-%d')}")
    print(f"- Fecha final  : {fecha_fin.strftime('%Y-%m-%d')}")
    print(f"- Total de datos: {total_datos:,}")

    # Test de normalidad (Shapiro-Wilk)
    stat, p = shapiro(df['valor'].dropna())
    print(f"📈 Test de normalidad (Shapiro-Wilk) - p-valor: {p:.4f}")

    return df
def Analisis_datos_valoresperdidos_caudal(df):
    """
    Analiza la frecuencia temporal, el porcentaje de datos perdidos y la existencia de datos nulos
    en una serie de tiempo de caudal. También identifica las fechas específicas que faltan.

    Parámetros:
    - df (DataFrame): debe contener una columna 'fecha' en formato datetime.
    """

    # Paso 1: Calcular la frecuencia temporal (intervalo más común entre observaciones)
    df['diferencia'] = df['fecha'].diff().dropna()
    frecuencia_promedio = df['diferencia'].mode()[0]  # Intervalo más común
    minutos_abs = int(abs(frecuencia_promedio.total_seconds() / 60))  # Convertido a minutos enteros

    # Paso 2: Calcular porcentaje de datos perdidos
    fecha_inicial = df['fecha'].min()
    fecha_final = df['fecha'].max()
    fechas_esperadas = pd.date_range(start=fecha_inicial, end=fecha_final, freq=f"{minutos_abs}min")
    conteo_esperado = len(fechas_esperadas)  # Total de timestamps esperados
    conteo_real = df['fecha'].nunique()      # Total de timestamps realmente observados
    datos_perdidos = conteo_esperado - conteo_real
    porcentaje_perdidos = (datos_perdidos / conteo_esperado) * 100

    # Paso 3: Calcular porcentaje de datos nulos
    porcentaje_nulos = df.isnull().sum() * 100 / len(df)

    # Mostrar resumen de estadísticas
    print("📉 Análisis de datos perdidos")
    print(f"- Intervalo dominante         : {minutos_abs} minutos")
    print(f"- Porcentaje de datos perdidos: {porcentaje_perdidos:.2f} %")
    print(f"- Porcentaje de fechas nulas  : {porcentaje_nulos['fecha']:.2f} %")

    # Paso 4: Identificar fechas faltantes
    fechas_reales = df['fecha'].dropna().sort_values().unique()
    fechas_reales_idx = pd.DatetimeIndex(fechas_reales)
    fechas_faltantes = fechas_esperadas.difference(fechas_reales_idx)

    # Mostrar resultados adicionales
    print(f"\n📆 Total de fechas esperadas: {len(fechas_esperadas):,}")
    print(f"✅ Total de fechas reales   : {len(fechas_reales):,}")
    print(f"❌ Total de fechas faltantes: {len(fechas_faltantes):,}")

    # Mostrar ejemplo de fechas faltantes
    if len(fechas_faltantes) > 0:
        print("\n🔎 Ejemplos de fechas perdidas:")
        print(fechas_faltantes[:10])
def Grafica_histograma_caudal(df, codigo, path):
    """
    Genera una figura con dos paneles:
    - Panel A: serie de tiempo original y suavizada (media móvil de 30 días)
    - Panel B: histograma con curva de densidad (KDE)

    Parámetros:
    - df: DataFrame con columnas ['fecha', 'valor']
    - codigo: nombre o código de la estación
    - path: ruta base donde se guardará la figura en una subcarpeta 'GRAFICOS'
    """

    # Estilo general de la visualización
    plt.style.use('seaborn-v0_8-white')
    sns.set_context("notebook", font_scale=1.1)

    # Suavizado con media móvil de 30 días
    df_sorted = df.sort_values('fecha')
    df_sorted['suavizado'] = df_sorted['valor'].rolling(window=30, min_periods=1).mean()

    # Estadísticas descriptivas
    media = df_sorted['valor'].mean()
    mediana = df_sorted['valor'].median()
    desviacion = df_sorted['valor'].std()

    # Crear figura con dos subgráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'hspace': 0.4})

    # --- Panel A: Serie de tiempo
    ax1.plot(df_sorted['fecha'], df_sorted['valor'], label='Original', color='blue', alpha=0.3)
    ax1.plot(df_sorted['fecha'], df_sorted['suavizado'], label='Suavizado (30 días)', color='#343A40', linewidth=1)
    ax1.set_title(f'A) Serie de tiempo - Estación {codigo}', fontsize=14, weight='bold', loc='left')
    ax1.set_ylabel('Valor')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Panel B: Histograma con KDE
    sns.histplot(df['valor'].dropna(), bins=30, kde=True, color='#ADB5BD',
                 ax=ax2, edgecolor='white', alpha=0.8)
    ax2.set_title('B) Distribución de los valores', fontsize=14, weight='bold', loc='left')
    ax2.set_xlabel('Valor')
    ax2.set_ylabel('Frecuencia')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Anotar estadísticas debajo del histograma
    stats_text = f"Media: {media:.1f}   |   Mediana: {mediana:.1f}   |   Desv. estándar: {desviacion:.1f}"
    ax2.annotate(stats_text, xy=(0.5, -0.25), xycoords='axes fraction',
                 fontsize=11, ha='center', style='italic')

    # Guardar figura
    plt.tight_layout()
    pathout = os.path.join(path, 'GRAFICOS', f'Grafica_histograma_caudal_{codigo}.png')
    os.makedirs(os.path.dirname(pathout), exist_ok=True)
    plt.savefig(pathout, dpi=300)
    plt.show()





                    
                    
                    











