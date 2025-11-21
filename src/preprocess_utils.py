import os
import pandas as pd
 
def load_data(path="../Data/raw/Crop Recommendation Dataset.xlsx"):
    """Carga el Excel y devuelve DataFrame."""
    return pd.read_excel(path)
 
def clean_data(df):
    """Limpia columnas, elimina duplicados y prepara el dataframe."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    return df
 
def categorize_crop(label):
    fruits = [
        "Apple", "Banana", "Grapes", "Guava", "Mango", "Orange",
        "Papaya", "Pomegranate", "Watermelon", "Muskmelon", "Litchi",
        "DragonFruit", "Jackfruit"
    ]

    vegetables = [
        "Cabbage", "Cauliflower", "Brinjal", "Tomato", "Potato", "Carrot",
        "Onion", "Garlic", "Ginger", "Spinach", "Broccoli", "Pumpkin",
        "Cucumber", "Beetroot", "Radish", "Capsicum", "Lettuce",
        "Green Beans", "French Beans", "Green Peas", "Ladys Finger",
        "Chinese Cabbage"
    ]

    cereals = ["Rice", "Maize", "Bajra", "Corn"]

    legumes = [
        "Pulses", "Chickpea", "Blackgram", "Kidneybeans",
        "Mothbeans", "Mungbeans", "Piegonpeas", "Rajma", "Soybean"
    ]

    spices = ["Turmeric", "Coriander"]

    commercial = [
        "Cotton", "Sugarcane", "Coffee", "Tea", "Arecanut", "Coconut",
        "Groundnut", "Mustard", "Walnuts", "Cashewnuts", "Poppy Seeds"
    ]

    medicinal = ["Aleovera", "Ashwagandha"]

    flowers = ["Marigold", "Rose"]

    if label in fruits:
        return "Fruit"
    elif label in vegetables:
        return "Vegetable"
    elif label in cereals:
        return "Cereal"
    elif label in legumes:
        return "Legume"
    elif label in spices:
        return "Spice"
    elif label in commercial:
        return "Commercial Crop"
    elif label in medicinal:
        return "Medicinal"
    elif label in flowers:
        return "Flower"
    else:
        return "Other"

# Aplicamos la categorizacion
def apply_categorization(df):
    """Aplica la recategorización al DataFrame (columna 'Category')."""
    # Asegurarse que la columna Label exista
    if "Label" not in df.columns:
        raise KeyError("La columna 'Label' no existe en el DataFrame.")
    df["Category"] = df["Label"].apply(categorize_crop)
    return df
 
def preprocess_data_df(df):
    """Pipeline que recibe un DataFrame ya cargado."""
    df = clean_data(df)
    df = apply_categorization(df)
    return df
 
def preprocess_data(path="../Data/raw/Crop Recommendation Dataset.xlsx"):
    """Pipeline que recibe una ruta: carga, limpia y categoriza."""
    df = load_data(path)
    df = clean_data(df)
    df = apply_categorization(df)
    return df
 
def save_processed(df, save_dir="../Data/processed/", filename="dataset_clean.csv"):
    os.makedirs(save_dir, exist_ok=True)
    fullpath = os.path.join(save_dir, filename)
    df.to_csv(fullpath, index=False)
    return fullpath
 
if __name__ == "__main__":
    df = preprocess_data()
    print("Primeras filas del dataset procesado:")
    print(df.head().to_string(index=False))
    category_counts = df["Category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]
    print("\nDistribución de categorías:")
    print(category_counts.to_string(index=False))
    out = save_processed(df)
    print("\nArchivo guardado en:", out)
