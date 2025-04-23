import pandas as pd

def load_data(file, date_col, target_col, delimiter=",", date_format="%Y-%m-%d"):
    df = pd.read_csv(file, delimiter=delimiter)
    df.rename(columns={date_col: "ds", target_col: "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"], format=date_format)
    return df

def select_features(df, features):
    return df[["ds", "y"] + features]