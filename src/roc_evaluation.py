import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import os

# Estilo científico y paleta accesible
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLOR_FULL = "#D55E00"   # Naranja/Rojo (Modelo Completo)
COLOR_SHORT = "#56B4E9"  # Azul (Formulario Corto)

def generar_curvas_roc():
    print("🚀 Iniciando Fase 7: Comparación de Modelos (ROC & AUC)...")
    
    # 1. Cargar Datos
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)
    os.makedirs("plots", exist_ok=True)

    # 2. Preparar los dos sets de datos
    y = df['Diabetes_binary']
    X_full = df.drop('Diabetes_binary', axis=1) # Las 21 variables
    X_short = df[['BMI', 'Age', 'GenHlth', 'Income', 'HighBP']] # Tu Top 5

    # 3. Dividir los datos (Mismo random_state para que la comparación sea justa)
    X_train_f, X_test_f, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    X_train_s, X_test_s, _, _ = train_test_split(X_short, y, test_size=0.2, random_state=42)

    # 4. Entrenar Modelo Completo
    print("   Entrenando Modelo Completo (21 variables)...")
    clf_full = LogisticRegression(max_iter=2000, random_state=42)
    clf_full.fit(X_train_f, y_train)
    # Extraer las probabilidades, no solo el 0 o 1
    probs_full = clf_full.predict_proba(X_test_f)[:, 1]

    # 5. Entrenar Modelo Corto
    print("   Entrenando Formulario Corto (5 variables)...")
    clf_short = LogisticRegression(max_iter=2000, random_state=42)
    clf_short.fit(X_train_s, y_train)
    probs_short = clf_short.predict_proba(X_test_s)[:, 1]

    # 6. Calcular métricas ROC
    fpr_f, tpr_f, _ = roc_curve(y_test, probs_full)
    auc_f = auc(fpr_f, tpr_f)

    fpr_s, tpr_s, _ = roc_curve(y_test, probs_short)
    auc_s = auc(fpr_s, tpr_s)

    # 7. Generar el Gráfico
    plt.figure(figsize=(9, 7))
    
    # Dibujar las curvas
    plt.plot(fpr_f, tpr_f, color=COLOR_FULL, lw=2.5, 
             label=f'Full Model (21 Variables) | AUC = {auc_f:.3f}')
    plt.plot(fpr_s, tpr_s, color=COLOR_SHORT, lw=2.5, linestyle='--', 
             label=f'Short Form (5 Variables) | AUC = {auc_s:.3f}')
    
    # Dibujar la línea de adivinanza (peor escenario)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=2, label='Random Guessing (AUC = 0.500)')

    # Estética
    plt.title('Diagnostic Power: Short Form vs. Full Model Evaluation', fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.legend(loc="lower right", frameon=False, fontsize=11)
    
    # Anotación estratégica
    perdida = (auc_f - auc_s)
    texto_roi = f"Efficiency Insight:\nReducing the survey by 76% (from 21 to 5 questions)\nresults in an AUC loss of only {perdida:.3f} points."
    plt.text(x=0.02, y=0.95, s=texto_roi, fontsize=10, style='italic', color='#333333',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'),
             verticalalignment='top')

    plt.tight_layout()
    plt.savefig("plots/6_roc_curve_comparison.png", dpi=300)
    plt.close()
    print("\n✅ Curva ROC generada exitosamente en: plots/6_roc_curve_comparison.png")

if __name__ == "__main__":
    generar_curvas_roc()