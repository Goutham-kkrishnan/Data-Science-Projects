import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """
    Cleans the raw dataframe, performs feature engineering, and encodes categorical variables.
    Returns:
        X (pd.DataFrame): feature matrix (all numeric)
        y (pd.Series): encoded target
        target_names (list): original class names for the target
    """
    df = df.copy()

    # ----- 1. Drop duplicates -----
    df.drop_duplicates(inplace=True)

    # ----- 2. Feature engineering (using original string columns) -----
    # BMI and Weight/Height ratio
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Wt_Ht_ratio'] = df['Weight'] / df['Height']

    # Active transport based on MTRANS (still string here)
    df['Active_transport'] = df['MTRANS'].isin(['Walking', 'Bike']).astype(int)

    # Activity score = FAF + Active_transport
    df['Activity_score'] = df['FAF'] + df['Active_transport']

    # Diet score: FCVC + NCP - (1 if FAVC == 'yes' else 0)
    df['Diet_score'] = df['FCVC'] + df['NCP'] - (df['FAVC'] == 'yes').astype(int)

    # Drop original Weight & Height
    df.drop(columns=['Weight', 'Height'], inplace=True)

    # ----- 3. Encode categorical columns -----
    # Identify object columns (including target)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Separate target before encoding others
    target_col = 'NObeyesdad'
    if target_col in cat_cols:
        cat_cols.remove(target_col)
        # Encode target
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col])
        target_names = le_target.classes_.tolist()
    else:
        target_names = None

    # Encode remaining categorical features
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # ----- 4. Separate features and target -----
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, target_names


def preprocess_features(df):
    """
    Feature engineering + categorical encoding
    (NO target handling here)
    """
    df = df.copy()

    # Feature engineering
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Wt_Ht_ratio'] = df['Weight'] / df['Height']
    df['Active_transport'] = df['MTRANS'].isin(['Walking', 'Bike']).astype(int)
    df['Activity_score'] = df['FAF'] + df['Active_transport']
    df['Diet_score'] = df['FCVC'] + df['NCP'] - (df['FAVC'] == 'yes').astype(int)

    # Drop original
    df.drop(columns=['Weight', 'Height'], inplace=True)

    return df
