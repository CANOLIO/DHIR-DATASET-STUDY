import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

sns.set_theme(style="whitegrid")

def seleccionar_mejores_variables():
    print("🌳 Entrenando modelo Random Forest para Selección de Características...")
    
    # 1. Cargar el dataset balanceado
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)

    # 2. Separar las "Preguntas" (X) del "Diagnóstico" (y)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']

    # 3. Entrenar el algoritmo Random Forest
    # n_estimators=100 significa que creará 100 árboles de decisión para votar
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_rf.fit(X, y)

    # 4. Extraer el "Feature Importance" (Importancia de cada variable)
    importancias = modelo_rf.feature_importances_
    
    df_importancias = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    print("\n✅ Top 10 Variables más predictivas según la IA:")
    print(df_importancias.head(10).to_string(index=False))

    # 5. Visualización Profesional
    plt.figure(figsize=(12, 8))
    
    # Graficamos el Top 15 para no saturar la imagen
    top_15 = df_importancias.head(15)
    sns.barplot(x='Importancia', y='Variable', data=top_15, palette='magma')
    
    plt.title('Top Predictors of Diabetes Risk (Random Forest Feature Importance)', fontsize=18, fontweight='bold')
    plt.xlabel('Relative Importance (Contribution to Model Accuracy)', fontsize=12)
    plt.ylabel('BRFSS Survey Variables', fontsize=12)
    plt.tight_layout()
    
    # Guardar la imagen
    os.makedirs("plots", exist_ok=True)
    ruta_guardado = "plots/feature_importance_rf.png"
    plt.savefig(ruta_guardado, dpi=300)
    print(f"\n📊 Gráfico guardado exitosamente en: {ruta_guardado}")

if __name__ == "__main__":
    seleccionar_mejores_variables()