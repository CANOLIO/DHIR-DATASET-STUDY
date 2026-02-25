import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Estilo más limpio y moderno
sns.set_theme(style="white")

def generar_mapa_calor_profesional():
    print("🎨 Generando Mapa de Calor Profesional...")
    
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)

    # 1. DICCIONARIO DE TRADUCCIÓN (Data Storytelling)
    # Seleccionamos solo las variables con mayor impacto biológico/social
    diccionario_nombres = {
        'Diabetes_binary': 'Diabetes Diagnosis (Target)',
        'GenHlth': 'Poor General Health',      # 5 es mala salud en la encuesta
        'HighBP': 'High Blood Pressure',
        'BMI': 'Body Mass Index (BMI)',
        'HighChol': 'High Cholesterol',
        'Age': 'Age Bracket',
        'DiffWalk': 'Difficulty Walking',
        'Income': 'Income Level',              # Factor protector (-)
        'Education': 'Education Level',        # Factor protector (-)
        'PhysActivity': 'Physical Activity',   # Factor protector (-)
        'HeartDiseaseorAttack': 'Heart Disease History',
        'PhysHlth': 'Poor Physical Health Days'
    }
    
    # 2. FILTRAR Y RENOMBRAR
    # Nos quedamos solo con las columnas del diccionario
    df_top = df[list(diccionario_nombres.keys())].rename(columns=diccionario_nombres)

    # 3. MATRIZ DE CORRELACIÓN
    corr = df_top.corr()

    # 4. CONFIGURACIÓN VISUAL
    # Un tamaño más horizontal y compacto
    plt.figure(figsize=(14, 10)) 
    
    # Máscara triangular
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 5. DIBUJAR EL HEATMAP
    # Usamos RdYlBu_r: Rojo (Riesgo/Positivo) y Azul (Protector/Negativo)
    heatmap = sns.heatmap(
        corr, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap="RdYlBu_r", 
        vmin=-0.5, vmax=0.5,  # 🔥 EL TRUCO: Comprimir la escala para que los colores exploten
        linewidths=1.5,       # Líneas de separación más gruesas
        cbar_kws={"shrink": .8, "label": "Correlation Strength"}
    )

    # 6. ESTÉTICA DE TEXTOS
    plt.title('Key Risk & Protective Factors for Diabetes', fontsize=22, pad=20, fontweight='bold', color='#2c3e50')
    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='500')
    plt.yticks(fontsize=11, fontweight='500')
    
    # Ajustar márgenes automáticamente para que no se corten los textos
    plt.tight_layout()

    # 7. GUARDAR
    ruta_guardado = "plots/correlation_matrix_professional.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(ruta_guardado, dpi=300)
    print(f"\n✅ Gráfico profesional guardado exitosamente en: {ruta_guardado}")

if __name__ == "__main__":
    generar_mapa_calor_profesional()