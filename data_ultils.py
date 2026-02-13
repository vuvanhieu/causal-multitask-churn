# data_ultils.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors


def balance_labels(y_array, target_dist=None):
    """
    Apply undersampling to balance the label distribution of y_array.
    If target_dist is None, equalizes all classes.
    """
    df = pd.DataFrame({'label': y_array})
    counts = df['label'].value_counts()
    min_count = target_dist if target_dist else counts.min()
    balanced_df = pd.concat([
        df[df['label'] == label].sample(n=min_count, replace=False, random_state=42)
        for label in counts.index
    ])
    return balanced_df.index.values


def apply_balancing_all_tasks(X_train, y_train_dict):
    """
    1. SMOTE cho task churn.
    2. Dùng NearestNeighbors để map lại label score, balance.
    3. Cân bằng lại score (multiclass) và balance (binary) bằng undersampling.
    """
    # Step 1: SMOTE for churn
    smote = SMOTE(random_state=42)
    X_res, y_churn_res = smote.fit_resample(X_train, y_train_dict['churn'])

    # Step 2: Align other labels
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, indices = nbrs.kneighbors(X_res)
    indices = indices.flatten()

    y_score_res = y_train_dict['score'][indices]
    y_balance_res = y_train_dict['balance'][indices]

    # Step 3: Balance score (multiclass) and balance (binary)
    score_labels = np.argmax(y_score_res, axis=1)
    idx_score_bal = balance_labels(score_labels)
    idx_balance_bal = balance_labels(y_balance_res)

    # Intersect all indices (churn-balanced already)
    final_idx = np.intersect1d(idx_score_bal, idx_balance_bal)

    # Final output
    return X_res[final_idx], {
        'churn': y_churn_res[final_idx],
        'score': y_score_res[final_idx],
        'balance': y_balance_res[final_idx]
    }


def apply_smote_to_main_task(X_train, y_train_dict):
    """
    Phiên bản nhẹ: chỉ SMOTE cho churn, align score & balance.
    """
    y_churn = y_train_dict['churn']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_churn)

    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, indices = nbrs.kneighbors(X_res)
    indices = indices.flatten()

    y_train_dict_new = {
        'churn': y_res,
        'score': y_train_dict['score'][indices],
        'balance': y_train_dict['balance'][indices]
    }
    return X_res, y_train_dict_new


def load_clean_data_multitask(filepath):
    """
    Load dữ liệu Bank Customer Churn, encode & chuẩn hóa.
    Trả về: X, y_dict, df_full
    """
    df = pd.read_csv(filepath)
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    df['CreditScoreClass'] = pd.cut(
        df['CreditScore'], bins=[0, 580, 700, 850], labels=[0, 1, 2]
    ).astype(int)
    df['HighBalanceFlag'] = (df['Balance'] > 100000).astype(int)

    X = df.drop(['Exited', 'CreditScoreClass', 'HighBalanceFlag'], axis=1).values
    X = StandardScaler().fit_transform(X)

    y_dict = {
        'churn': df['Exited'].values,
        'score': to_categorical(df['CreditScoreClass'].values, num_classes=3),
        'balance': df['HighBalanceFlag'].values
    }

    return X, y_dict, df
