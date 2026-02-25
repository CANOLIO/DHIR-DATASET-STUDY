import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. ESTILO CIENTÍFICO (Nature / The Lancet Style)
# Fondo blanco puro, sin cuadrículas invasivas, tipografía sans-serif limpia
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 2. PALETA ACCESIBLE (Colorblind-friendly / Okabe-Ito inspired)
# Azul (Sano) y Naranja/Bermellón (Diabético)
COLOR_SANO = "#56B4E9" 
COLOR_DIABETES = "#D55E00"
paleta_accesible = [COLOR_SANO, COLOR_DIABETES]

def generar_analisis_demografico():
    print("🔬 Iniciando Fase 4 y 5: Análisis Demográfico con Diseño Científico...")
    
    # Cargar datos
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)

    df['Diagnosis'] = df['Diabetes_binary'].map({
        0.0: 'Healthy Control (Blue)', 
        1.0: 'Diabetic Group (Orange)'
    })
    os.makedirs("plots", exist_ok=True)

    # ==========================================
    # GRÁFICO 1: BMI (Distribución Fisiológica)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Usamos un KDE (Apagamos la leyenda automática con legend=False para controlarla nosotros)
    sns.kdeplot(data=df, x='BMI', hue='Diagnosis', fill=True, 
                palette=paleta_accesible, alpha=0.6, linewidth=2, common_norm=False, legend=False)
    
    # Línea clínica de obesidad (BMI = 30)
    plt.axvline(30, color='black', linestyle='--', alpha=0.5)
    
    plt.title('Metabolic Shift: BMI Distribution in Healthy vs. Diabetic Populations', fontweight='bold', pad=15)
    plt.xlabel('Body Mass Index (BMI)')
    plt.ylabel('Density of Population')
    plt.xlim(15, 60) # Cortamos extremos irreales para enfocar la vista
    
    # --- NUEVO: LEYENDA MANUAL Y A PRUEBA DE BALAS ---
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    # Creamos los "cuadritos" de color y la línea de muestra
    sano_patch = mpatches.Patch(color=COLOR_SANO, alpha=0.6, label='Healthy Control (Blue)')
    diab_patch = mpatches.Patch(color=COLOR_DIABETES, alpha=0.6, label='Diabetic Group (Orange)')
    obes_line = mlines.Line2D([], [], color='black', linestyle='--', alpha=0.5, label='Obesity Threshold (BMI 30)')
    
    # Le pasamos nuestras creaciones exactas a la leyenda
    plt.legend(handles=[sano_patch, diab_patch, obes_line], frameon=False, loc='upper right')
    
    plt.tight_layout()
    plt.savefig("plots/1_demographics_bmi.png", dpi=300)
    plt.close()
    print("✅ Gráfico de BMI generado (con leyenda arreglada).")

    # ==========================================
    # GRÁFICO 2: EDAD (Evolución Temporal del Riesgo)
    # ==========================================
    # Mapeo de categorías del BRFSS a edades reales
    age_map = {1:'18-24', 2:'25-29', 3:'30-34', 4:'35-39', 5:'40-44', 6:'45-49', 
               7:'50-54', 8:'55-59', 9:'60-64', 10:'65-69', 11:'70-74', 12:'75-79', 13:'80+'}
    df['Age_Group'] = df['Age'].map(age_map)
    
    # Calculamos % de diabéticos por grupo
    age_risk = df.groupby('Age_Group')['Diabetes_binary'].mean() * 100
    
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x=age_risk.index, y=age_risk.values, color=COLOR_DIABETES, alpha=0.85)
    
    plt.title('Age as a Risk Factor: Diabetes Prevalence by Age Group', fontweight='bold', pad=15)
    plt.xlabel('Age Bracket')
    plt.ylabel('Prevalence Rate (%)')
    plt.xticks(rotation=45)
    
    # Añadir los porcentajes encima de cada barra
    for p in bars.patches:
        bars.annotate(f'{p.get_height():.1f}%', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', fontsize=9, xytext=(0, 5), textcoords='offset points')
        
    plt.tight_layout()
    plt.savefig("plots/2_demographics_age.png", dpi=300)
    plt.close()
    print("✅ Gráfico de Edad generado.")

    # ==========================================
    # GRÁFICO 3: INGRESOS (Determinantes Sociales)
    # ==========================================
    # Mapeo de categorías del BRFSS a ingresos
    income_map = {1:'< $10k', 2:'$10k-$15k', 3:'$15k-$20k', 4:'$20k-$25k', 
                  5:'$25k-$35k', 6:'$35k-$50k', 7:'$50k-$75k', 8:'> $75k'}
    df['Income_Level'] = df['Income'].map(income_map)
    
    income_risk = df.groupby('Income_Level')['Diabetes_binary'].mean() * 100
    # Ordenamos el índice para que tenga sentido lógico
    orden_ingresos = list(income_map.values())
    
    plt.figure(figsize=(10, 6))
    sns.pointplot(x=income_risk.index, y=income_risk.values, order=orden_ingresos, 
                  color=COLOR_SANO, scale=1.2, markers="o")
    
    # Rellenamos el área bajo la curva para darle impacto visual
    plt.fill_between(range(len(orden_ingresos)), income_risk[orden_ingresos], color=COLOR_SANO, alpha=0.2)
    
    plt.title('Social Determinants: The Protective Effect of Higher Income', fontweight='bold', pad=15)
    plt.xlabel('Annual Household Income (USD)')
    plt.ylabel('Prevalence Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, max(income_risk)+10) # Dar aire arriba
    
    texto_conclusion = "Clinical Insight:\nA ~50% reduction in prevalence is observed\nas income moves from <$10k to >$75k."
    plt.text(x=3.5, y=60, s=texto_conclusion, 
             fontsize=11, style='italic', color='#333333',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.6'))
    
    plt.tight_layout()
    plt.savefig("plots/3_demographics_income.png", dpi=300)
    plt.close()
    print("✅ Gráfico de Ingresos generado (con tarjeta clínica).")
    print("\n🎉 ¡Todos los gráficos de calidad científica fueron guardados en la carpeta 'plots/'!")

if __name__ == "__main__":
    generar_analisis_demografico()