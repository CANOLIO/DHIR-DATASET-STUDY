import pandas as pd
import os

def auditar_dataset(ruta_archivo):
    """
    Función que actúa como un 'escáner' para auditar la calidad 
    de cualquier dataset en formato CSV.
    """
    nombre_archivo = os.path.basename(ruta_archivo)
    
    print(f"\n{'='*60}")
    print(f"🔬 REPORTE DE AUDITORÍA: {nombre_archivo}")
    print(f"{'='*60}")
    
    try:
        # Cargar el dataset
        df = pd.read_csv(ruta_archivo)
        
        # 1. Dimensiones (Filas y Columnas)
        print("\n📊 1. DIMENSIONES DEL DATASET:")
        print(f"   - Total de registros (filas): {df.shape[0]:,}")
        print(f"   - Total de variables (columnas): {df.shape[1]}")
        
        # 2. Valores Nulos
        nulos_totales = df.isnull().sum().sum()
        print(f"\n⚠️ 2. INTEGRIDAD DE DATOS (NULOS):")
        print(f"   - Valores nulos totales: {nulos_totales}")
        if nulos_totales > 0:
            print("   - Detalle de nulos por columna:")
            nulos_por_columna = df.isnull().sum()
            print(nulos_por_columna[nulos_por_columna > 0])
            
        # 3. Filas Duplicadas
        duplicados = df.duplicated().sum()
        porcentaje_duplicados = (duplicados / df.shape[0]) * 100
        print(f"\n🔄 3. REGISTROS DUPLICADOS:")
        print(f"   - Filas idénticas: {duplicados:,} ({porcentaje_duplicados:.2f}% de la muestra)")
        
        # 4. Tipos de Datos (Vistazo general)
        print(f"\n📝 4. TIPOS DE DATOS DETECTADOS:")
        conteo_tipos = df.dtypes.value_counts()
        for tipo, cantidad in conteo_tipos.items():
            print(f"   - {tipo}: {cantidad} columnas")
            
        # 5. Muestra de las primeras 3 filas (para entender el formato)
        print(f"\n👀 5. VISTAZO A LOS DATOS (Primeras 3 filas):")
        # Imprimimos solo las primeras 6 columnas para no colapsar la terminal
        print(df.iloc[:3, :6].to_string())
        
        print(f"\n{'='*60}\n")
        return df

    except FileNotFoundError:
        print(f"❌ ERROR CRÍTICO: No se encontró el archivo en la ruta especificada.")
        print(f"   Ruta buscada: {ruta_archivo}")
        return None

if __name__ == "__main__":
    # Definimos la ruta exacta de tu proyecto
    ruta_base = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET"
    
    # Archivo 1: El dataset balanceado (Nuestro punto de partida estratégico)
    archivo_5050 = os.path.join(ruta_base, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    
    # Ejecutamos la auditoría
    df_balanceado = auditar_dataset(archivo_5050)
    
    archivo_incompleto = os.path.join(ruta_base, "diabetes_binary_health_indicators_BRFSS2015.csv")
    df_incompleto = auditar_dataset(archivo_incompleto)