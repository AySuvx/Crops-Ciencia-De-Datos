import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
 


def load_clean_dataset(path="C:/Users/ACER/Documents/SEPTIMO SEMESTRE/Ciencia de datos/Crops-Ciencia-De-Datos/Data/processed/dataset_clean.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERROR: No existe el archivo en: {path}")
    return pd.read_csv(path)
   

def split_xy(df):
    X = df[["Temperature", "Humidity", "pH", "Rainfall"]]
    y = df["Category"]
    return X, y
 
#MODELOS 

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def train_svm(X_train, y_train):
    svm = SVC(C=1, gamma="scale", kernel="rbf")
    svm.fit(X_train, y_train)
    return svm
 
 

def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    report = classification_report(y_test, model.predict(X_test))
    cm_norm = confusion_matrix(
        y_test,
        model.predict(X_test),
        normalize="true"
    )

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "report": report,
        "cm": cm_norm
    }
 
 

def show_confusion_matrix(cm, title, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()
  

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath, compress=("lzma", 9))
  

def train_all_models():

    print("\nCargando dataset...\n")
    df = load_clean_dataset()

    X, y = split_xy(df)

    labels = sorted(df["Category"].unique())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = {}

    #RANDOM FOREST
    print("Entrenando Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_eval = evaluate_model(rf, X_train, X_test, y_train, y_test)
    print("\nReporte RF:\n", rf_eval["report"])
    show_confusion_matrix(rf_eval["cm"], "Random Forest - Categoría", labels)
    save_model(rf, "../Models/category_RF_model.joblib")
    results["RF"] = rf_eval

    #KNN
    print("Entrenando KNN...")
    knn = train_knn(X_train, y_train)
    knn_eval = evaluate_model(knn, X_train, X_test, y_train, y_test)
    print("\nReporte KNN:\n", knn_eval["report"])
    show_confusion_matrix(knn_eval["cm"], "KNN - Categoría", labels)
    save_model(knn, "../Models/category_KNN_model.joblib")
    results["KNN"] = knn_eval

    # === SVM ===
    print("Entrenando SVM...")
    svm = train_svm(X_train, y_train)
    svm_eval = evaluate_model(svm, X_train, X_test, y_train, y_test)
    print("\nReporte SVM:\n", svm_eval["report"])
    show_confusion_matrix(svm_eval["cm"], "SVM - Categoría", labels)
    save_model(svm, "../Models/category_SVM_model.joblib")
    results["SVM"] = svm_eval

    return results
 
#EN ESTA SECCION HACEMOS LA COMPARACIÓN ENTRE MODELOS PARA DETERMINAR EL MEJOR

def compare_models(results):
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    print("COMPARACION FINAL DE MODELOS") 

    comparison = []

    for model_name, metrics in results.items():
        comparison.append({
            "Modelo": model_name,
            "Accuracy Entrenamiento": round(metrics["train_acc"], 4),
            "Accuracy Prueba": round(metrics["test_acc"], 4)
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df)

    # Seleccionar mejor modelo
    best_row = comp_df.loc[comp_df["Accuracy Prueba"].idxmax()]
    best_model_name = best_row["Modelo"]
    best_acc = best_row["Accuracy Prueba"]

    print("\nMEJOR MODELO:")
    print("Modelo:", best_model_name)
    print("Accuracy de prueba:", best_acc)
 
    #GRÁFICOS 

    sns.set(style="whitegrid")

    #Accuracy de entrenamiento
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Modelo", y="Accuracy Entrenamiento", data=comp_df)
    plt.title("Comparación: Accuracy de Entrenamiento")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    #Accuracy de prueba
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Modelo", y="Accuracy Prueba", data=comp_df)
    plt.title("Comparación: Accuracy de Prueba")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # --- Comparativo general ---
    plt.figure(figsize=(10, 6))
    comp_df_melt = comp_df.melt(
        id_vars="Modelo",
        value_vars=["Accuracy Entrenamiento", "Accuracy Prueba"],
        var_name="Tipo",
        value_name="Accuracy"
    )
    sns.barplot(x="Modelo", y="Accuracy", hue="Tipo", data=comp_df_melt)
    plt.title("Comparación General de Modelos")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return comp_df, best_model_name



 

if __name__ == "__main__":
    print("\nEntrenando modelos para Categoría...\n")
    results = train_all_models()
    comp_df, best_model = compare_models(results)
    print("\nProceso completado.\n")
