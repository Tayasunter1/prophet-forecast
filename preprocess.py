import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    df.rename(columns={df.columns[0]:'ds', df.columns[1]:'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds']) # Ensure date format
    return df

def select_features(df, features):
    return df[["ds", "y"] + features]