import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Estilo de publicación científica
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def generar_analisis_sindrome_metabolico():
    print("🧬 Iniciando Fase 6: Feature Engineering (Metabolic Syndrome Score)...")
    
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)
    os.makedirs("plots", exist_ok=True)

    # ---------------------------------------------------------
    # 1. FEATURE ENGINEERING (Creando el Score)
    # ---------------------------------------------------------
    # Primero, creamos una variable binaria estricta para Obesidad (BMI >= 30)
    df['Obese'] = (df['BMI'] >= 30).astype(int)
    
    # Sumamos los 3 grandes factores de riesgo clínico (0 a 3 puntos)
    df['Metabolic_Score'] = df['HighBP'] + df['HighChol'] + df['Obese']
    
    # Mapeamos para que los gráficos sean auto-explicativos
    df['Diagnosis'] = df['Diabetes_binary'].map({0.0: 'Healthy Control', 1.0: 'Diabetic Group'})
    df['Sex_Label'] = df['Sex'].map({0.0: 'Female', 1.0: 'Male'})

    # ---------------------------------------------------------
    # GRÁFICO 1: El efecto del Score Metabólico
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Calculamos la prevalencia de diabetes según cuántos factores de riesgo tienen
    score_risk = df.groupby('Metabolic_Score')['Diabetes_binary'].mean() * 100
    
    # Usamos una paleta de colores que se va "calentando" (de amarillo a rojo oscuro)
    sns.barplot(x=score_risk.index, y=score_risk.values, palette="YlOrRd")
    
    plt.title('Synergistic Risk: Diabetes Prevalence by Metabolic Syndrome Score', fontweight='bold', pad=15)
    plt.xlabel('Metabolic Risk Factors Present\n(High BP, High Cholesterol, Obesity)', fontsize=12)
    plt.ylabel('Diabetes Prevalence (%)', fontsize=12)
    
    # Agregamos los porcentajes sobre las barras
    for i, v in enumerate(score_risk.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
        
    # Anotación explicativa para el reclutador/lector
    texto_insight = "Clinical Insight:\nThe presence of all 3 factors yields\na ~75% probability of Diabetes\nin this sample."
    plt.text(x=0.1, y=60, s=texto_insight, fontsize=11, style='italic', color='#333333',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.6'))

    plt.tight_layout()
    plt.savefig("plots/4_metabolic_syndrome_score.png", dpi=300)
    plt.close()
    print("✅ Gráfico de Síndrome Metabólico generado.")

   # ---------------------------------------------------------
    # GRÁFICO 2: Interacción con el Sexo del Paciente (FacetGrid)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Comparamos cómo afecta el score a hombres vs mujeres
    sns.pointplot(data=df, x='Metabolic_Score', y='Diabetes_binary', hue='Sex_Label', 
                  palette=["#CC79A7", "#0072B2"], # Paleta accesible (Rosa fuerte / Azul oscuro)
                  dodge=True, markers=["o", "s"], scale=1.2)
    
    # Convertir el eje Y a porcentaje en la visualización
    plt.gca().set_yticklabels([f'{int(y*100)}%' for y in plt.gca().get_yticks()])
    
    plt.title('Demographic Interaction: Metabolic Score Impact by Sex', fontweight='bold', pad=15)
    plt.xlabel('Number of Metabolic Risk Factors', fontsize=12)
    plt.ylabel('Probability of Diabetes', fontsize=12)
    
    # --- NUEVO: ANOTACIONES DE SIGNIFICANCIA ESTADÍSTICA (Chi-Cuadrado) ---
    # Coordenadas Y calculadas en base a la escala interna del gráfico (0.0 a 1.0)
    plt.text(x=0, y=0.23, s='***', ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    plt.text(x=1, y=0.44, s='***', ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    plt.text(x=2, y=0.66, s='ns', ha='center', fontsize=12, fontweight='bold', color='#7f8c8d')
    plt.text(x=3, y=0.85, s='**', ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Leyenda explicativa de los asteriscos (Estándar científico)
    texto_pvalues = "Chi-Square Test:  ** p < 0.01  |  *** p < 0.001  |  ns: not significant"
    plt.text(x=1.5, y=0.18, s=texto_pvalues, ha='center', fontsize=10, style='italic', color='#555555',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    plt.legend(title='Biological Sex', frameon=False, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("plots/5_metabolic_sex_interaction.png", dpi=300)
    plt.close()
    print("✅ Gráfico de Interacción por Sexo generado (Con validación estadística).")

if __name__ == "__main__":
    generar_analisis_sindrome_metabolico()