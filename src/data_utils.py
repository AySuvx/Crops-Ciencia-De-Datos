import pandas as pd

def load_excel(path):
    df = pd.read_excel(path)
    return df

def save_csv(df, path):
    df.to_csv(path, index=False)
    return path
