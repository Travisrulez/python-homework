"""Обучение модели стоимости квартир по сокращённому realty_data.csv.

Использование: python train_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'realty_train.csv'
MODEL_PATH = BASE / 'model.joblib'
META_PATH = BASE / 'model_info.json'
FEATURES = ['city', 'total_square', 'rooms', 'floor']
RANDOM_STATE = 42


def load_and_prepare(path: Path = DATA_PATH) -> pd.DataFrame:
    """Читаем компактный датасет и убираем объявления за пределами рабочего диапазона."""
    df = pd.read_csv(path, usecols=['price', *FEATURES], low_memory=False)
    for col in ['price', 'total_square', 'rooms', 'floor']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['city'] = df['city'].astype('string').str.strip()
    df = df.dropna(subset=['price', 'total_square', 'rooms', 'floor', 'city'])
    df = df[df['city'] != '']
    df = df[
        df['price'].between(2_000_000, 250_000_000)
        & df['total_square'].between(15, 300)
        & df['rooms'].between(0, 7)
        & df['floor'].between(1, 50)
    ].copy()
    df['city'] = df['city'].astype(str)
    df = df.drop_duplicates()
    if len(df) < 100:
        raise ValueError('Недостаточно корректных строк для обучения модели')
    return df


def make_model():
    encoder = ColumnTransformer(
        transformers=[
            ('city', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['city']),
            ('numeric', 'passthrough', ['total_square', 'rooms', 'floor']),
        ],
        remainder='drop',
    )
    regressor = Pipeline([
        ('preprocess', encoder),
        ('predictor', HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=210,
            max_leaf_nodes=31,
            min_samples_leaf=24,
            l2_regularization=0.5,
            random_state=RANDOM_STATE,
        )),
    ])
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def train_and_evaluate(path: Path = DATA_PATH, save: bool = True):
    df = load_and_prepare(path)
    X = df[FEATURES]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline = DummyRegressor(strategy='median')
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'n_records': int(len(df)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'mae_rub': round(float(mean_absolute_error(y_test, pred))),
        'mape_percent': round(float(mean_absolute_percentage_error(y_test, pred) * 100), 2),
        'r2': round(float(r2_score(y_test, pred)), 4),
        'baseline_mae_rub': round(float(mean_absolute_error(y_test, baseline_pred))),
        'baseline_r2': round(float(r2_score(y_test, baseline_pred)), 4),
        'cities': sorted(df['city'].unique().tolist()),
        'random_state': RANDOM_STATE,
        'features': FEATURES,
        'target': 'price',
        'data_source': 'realty_data.csv (пользовательский датасет объявлений, Москва и область)',
    }
    if save:
        joblib.dump(model, MODEL_PATH, compress=3)
        META_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    return model, metrics


if __name__ == '__main__':
    _, info = train_and_evaluate()
    print('Обучение успешно завершено')
    print(f"Объектов: {info['n_records']} (train={info['n_train']}; test={info['n_test']})")
    print(f"MAE модели: {info['mae_rub']:,} руб.; MAE бейзлайна: {info['baseline_mae_rub']:,} руб.")
    print(f"MAPE: {info['mape_percent']}%; R²: {info['r2']}")
    print('Модель сохранена в:', MODEL_PATH)
