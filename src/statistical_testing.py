import pandas as pd
from scipy.stats import chi2_contingency

def test_estadistico():
    print("🧮 Iniciando Validación Estadística (Prueba Chi-Cuadrado)...")
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)
    
    # Recrear el score
    df['Obese'] = (df['BMI'] >= 30).astype(int)
    df['Metabolic_Score'] = df['HighBP'] + df['HighChol'] + df['Obese']
    
    print("\n📊 Hipótesis: ¿Existe diferencia estadísticamente significativa entre hombres y mujeres?")
    print("-" * 75)
    
    # Evaluar estadísticamente cada punto del gráfico
    for score in range(4):
        subset = df[df['Metabolic_Score'] == score]
        
        # Crear tabla de contingencia (Frecuencias de Sanos vs Diabéticos por Sexo)
        contingency_table = pd.crosstab(subset['Sex'], subset['Diabetes_binary'])
        
        # Calcular estadístico
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Interpretar el rigor científico (p < 0.05)
        if p_value < 0.05:
            resultado = "✅ SÍ (Diferencia Real)"
        else:
            resultado = "❌ NO (Posible Azar)"
            
        # Formatear el p-value para que no salga en notación científica si es muy pequeño
        p_formateado = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
        
        print(f"Factores Metabólicos [{score}]: Valor-P = {p_formateado:<8} | Significativo: {resultado}")

if __name__ == "__main__":
    test_estadistico()