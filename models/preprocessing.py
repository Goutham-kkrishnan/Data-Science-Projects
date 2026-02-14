import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):   
    df = df.copy()

    df.drop_duplicates(inplace=True)

    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Wt_Ht_ratio'] = df['Weight'] / df['Height']

    df['Active_transport'] = df['MTRANS'].isin(['Walking', 'Bike']).astype(int)

    df['Activity_score'] = df['FAF'] + df['Active_transport']

    df['Diet_score'] = df['FCVC'] + df['NCP'] - (df['FAVC'] == 'yes').astype(int)

    df.drop(columns=['Weight', 'Height'], inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    target_col = 'NObeyesdad'
    if target_col in cat_cols:
        cat_cols.remove(target_col)
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])
        target_names = le_target.classes_.tolist()
    else:
        target_names = None

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, target_names


def preprocess_features(df):
    """
    Feature engineering + categorical encoding
    (NO target handling here)
    """
    df = df.copy()

    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Wt_Ht_ratio'] = df['Weight'] / df['Height']
    df['Active_transport'] = df['MTRANS'].isin(['Walking', 'Bike']).astype(int)
    df['Activity_score'] = df['FAF'] + df['Active_transport']
    df['Diet_score'] = df['FCVC'] + df['NCP'] - (df['FAVC'] == 'yes').astype(int)

    df.drop(columns=['Weight', 'Height'], inplace=True)

    return df
