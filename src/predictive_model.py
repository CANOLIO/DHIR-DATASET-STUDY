import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

sns.set_theme(style="white")

def entrenar_modelo_corto():
    print("🚀 Iniciando Fase 3: Entrenamiento del Modelo Predictivo...")
    
    # 1. Cargar el dataset
    ruta = "/Users/fabian/Downloads/Diabetes_DATASET/DATASET/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    df = pd.read_csv(ruta)

    # 2. Seleccionar SOLO el "Formulario Corto" (Top 5 variables de la Fase 2)
    # Esto demuestra que podemos ahorrarle al paciente 16 preguntas innecesarias.
    top_features = ['BMI', 'Age', 'GenHlth', 'Income', 'HighBP']
    X = df[top_features]
    y = df['Diabetes_binary']

    # 3. Dividir los datos (Examen de Entrenamiento vs Examen Final)
    # Guardamos el 20% de los pacientes (14,000 aprox) para probar si el modelo realmente aprendió
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   - Pacientes para entrenar a la IA: {len(X_train):,}")
    print(f"   - Pacientes nuevos para probarla: {len(X_test):,}")

    # 4. Entrenar el Modelo (Regresión Logística Clínica)
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)

    # 5. Hacer que el modelo diagnostique a los pacientes nuevos (Test)
    y_pred = modelo.predict(X_test)

    # 6. Evaluar los resultados
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🏆 RESULTADOS DEL DIAGNÓSTICO IA (FORMULARIO CORTO)")
    print("="*50)
    print(f"Precisión General (Accuracy): {accuracy * 100:.2f}%\n")
    print("Reporte Clínico Detallado:")
    print(classification_report(y_test, y_pred, target_names=['Sano (0)', 'Diabético (1)']))
    
    # 7. Graficar la Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Mapa de calor para la matriz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicción: Sano', 'Predicción: Diabético'],
                yticklabels=['Realidad: Sano', 'Realidad: Diabético'],
                annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Clinical Confusion Matrix (Short Form Prediction)', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    
    # Guardar gráfico
    os.makedirs("plots", exist_ok=True)
    ruta_guardado = "plots/confusion_matrix_model.png"
    plt.savefig(ruta_guardado, dpi=300)
    print(f"\n📊 Matriz de Confusión guardada en: {ruta_guardado}")

if __name__ == "__main__":
    entrenar_modelo_corto()