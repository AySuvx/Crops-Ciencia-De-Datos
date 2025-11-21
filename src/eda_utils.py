import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
import math
 
def load_clean_data(path="../Data/processed/dataset_clean.csv"):
    return pd.read_csv(path)
 
def show_basic_info(df):
    display(Markdown("### Primeras 5 filas"))
    display(df.head())

    display(Markdown("### Estadísticas descriptivas"))
    display(df.describe().T)

    display(Markdown("### Conteo por categoría"))
    display(df["Category"].value_counts())
 
def plot_histograms(df, numeric_cols):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribución de {col}")
    plt.tight_layout()
    plt.show()
 
def plot_boxplots(df, numeric_cols):
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.show()
 
def plot_correlation(df, numeric_cols):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.show()
 
def plot_category_boxplots(df, numeric_cols):
    n_cols = 2
    n_rows = math.ceil(len(numeric_cols) / n_cols)

    plt.figure(figsize=(14, 5 * n_rows))

    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x=df["Category"], y=df[col])
        plt.xticks(rotation=45)
        plt.title(f"{col} por categoría")

    plt.tight_layout()
    plt.show()
 
def plot_pairplot(df, numeric_cols):
    sns.pairplot(df, hue="Category", vars=numeric_cols)
    plt.show()
