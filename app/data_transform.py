import numpy as np
import pandas as pd

from SETTINGS import FIRST_DATE


def reduce_data(df):
    # df.drop_duplicates(inplace=True)
    # df['capacity_bytes'] = (df['capacity_bytes'] / 1000000000000).astype(np.int64) # to terabytes
    df = df.drop(df.filter(regex='normalized$').columns, axis=1)  # raw
    num_features = df.select_dtypes(include="number")
    for el in num_features:
        for n_type in (np.int32, np.int16, np.int8):
            if df[el].isna().sum() == 0:
                if (df[el] == df[el].astype(n_type)).sum() == len(df):
                    df[el] = df[el].astype(n_type)
    return df


def append_disc_type(df):
    df['type'] = df['model'].apply(lambda x: 1 if 'SSD' in x else 0)
    return df


def append_period_col(df):
    """add days between 1st date of df and last date of disc in df"""
    df['date'] = pd.to_datetime(df['date'])
    first_date = pd.to_datetime(FIRST_DATE)
    df['days_between'] = df.groupby('serial_number')['date'].transform(
        lambda x: (x.max() - first_date).days
    )
    # print(df.columns)
    return df
