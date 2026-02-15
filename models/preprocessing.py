import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
TARGET_MAPPING = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
}

TARGET_NAMES = list(TARGET_MAPPING.keys())

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
    # if target_col in cat_cols:
    #     cat_cols.remove(target_col)
    #     le_target = LabelEncoder()
    #     df[target_col] = le_target.fit_transform(df[target_col])
    #     target_names = le_target.classes_.tolist()
    # else:
    #     target_names = None
    df['NObeyesdad'] = df['NObeyesdad'].map(TARGET_MAPPING)

    df= encode_categorical_columns(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, TARGET_NAMES


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

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns using predefined mapping dictionaries.
    Returns a NEW encoded DataFrame (original df is not modified).
    """

    encoding_maps = {
        "Gender": {
            "Female": 0, "Male": 1
        },
        "family_history_with_overweight": {
            "no": 0, "yes": 1
        },
        "FAVC": {
            "no": 0, "yes": 1
        },
        "SMOKE": {
            "no": 0, "yes": 1
        },
        "SCC": {
            "no": 0, "yes": 1
        },
        "CAEC": {
            "no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3
        },
        "CALC": {
            "no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3
        },
        "MTRANS": {
            "Automobile": 0,
            "Motorbike": 1,
            "Public_Transportation": 2,
            "Bike": 3,
            "Walking": 4
        }
    }

    df_encoded = df.copy()

    for col, mapping in encoding_maps.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

            # Safety check for unseen categories
            if df_encoded[col].isna().any():
                unseen = set(df[col].unique()) - set(mapping.keys())
                raise ValueError(
                    f"Unseen categories in column '{col}': {unseen}"
                )

    return df_encoded