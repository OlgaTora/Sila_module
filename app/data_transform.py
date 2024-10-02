import numpy as np
import pandas as pd

from SETTINGS import FIRST_DATE, MODELS


def reduce_data(df):
    """Удаление столбцов с нормализованными данными и приведение их к типу float"""
    df.drop_duplicates(inplace=True)
    df = df.drop(df.filter(regex="normalized$").columns, axis=1)  # raw
    num_features = df.select_dtypes(include="number")
    for col in num_features:
        df[col] = df[col].astype(np.float64)
    return df


def append_disc_type(df):
    """Добавление типа диска на основании данных, собранных из интернета."""
    models = pd.read_csv(MODELS)
    df = df.merge(models, on='model')
    df['type'].fillna(df['type'].mode())
    return df


def append_period_col(df):
    """Функция добавляющая колонку с периодом между датой начала исследования и датой
    последней записи по каждому диску"""
    df["date"] = pd.to_datetime(df["date"])
    first_date = pd.to_datetime(FIRST_DATE)
    df["days_between"] = df.groupby("serial_number")["date"].transform(
        lambda x: (x.max() - x.min()).days if x.max() != x.min() else (x.max() - first_date).days
    )
    return df
