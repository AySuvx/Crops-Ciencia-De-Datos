import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, class_names):
    """
    Evalúa un modelo ya entrenado e imprime:
    - Accuracy
    - Classification Report
    - Matriz de confusión
    Además genera:
    - Heatmap de matriz de confusión
    - Dispersión real vs predicho
    """

    # -----------------------------
    # 1. Predicciones
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # 2. Métricas
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print("=================================")
    print(" ACCURACY DEL MODELO")
    print("=================================")
    print("Accuracy:", accuracy)
    print("\n")

    print("=================================")
    print(" CLASSIFICATION REPORT")
    print("=================================")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # 3. Matriz de confusión
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Matriz de Confusión Normalizada")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Gráfico Real vs Predicho
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label="Real", alpha=0.7)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicción", alpha=0.7)
    plt.title("Comparación entre valores reales y predichos")
    plt.xlabel("Índice de muestra")
    plt.ylabel("Categoría")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return accuracy, cm
