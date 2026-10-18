"""Домашнее задание № 10: REST API оценки недвижимости на FastAPI.
Swagger UI: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.joblib"
INFO_FILE = BASE_DIR / "model_info.json"

app = FastAPI(
    title="Прогноз стоимости недвижимости | HW 10",
    description=(
        "Учебный REST API для определения ориентировочной стоимости квартиры "
    ),
    version="1.0.0",
)


class ApartmentFeatures(BaseModel):
    """Четыре входных признака — как в модели предыдущего ДЗ."""

    city: str = Field("Москва", min_length=1, description="Город из списка доступных городов", examples=["Москва"])
    total_square: float = Field(60.0, ge=15, le=300, description="Площадь квартиры, м²", examples=[60.0])
    rooms: int = Field(2, ge=0, le=7, description="Количество комнат (0 = студия)", examples=[2])
    floor: int = Field(5, ge=1, le=50, description="Этаж квартиры", examples=[5])


class PredictionResponse(BaseModel):
    predicted_price_rub: float = Field(description="Прогнозируемая стоимость квартиры, руб.")
    price_per_sqm_rub: float = Field(description="Ориентировочная стоимость 1 м², руб.")
    currency: str = "RUB"
    features: ApartmentFeatures


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@lru_cache(maxsize=1)
def load_resources():
    """Загружаем модель ровно один раз на процесс."""
    model = joblib.load(MODEL_FILE)
    info = json.loads(INFO_FILE.read_text(encoding="utf-8"))
    return model, info


def make_prediction(features: ApartmentFeatures) -> PredictionResponse:
    """Единый расчёт для GET и POST, чтобы прогнозы совпадали."""
    try:
        model, info = load_resources()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Модель временно недоступна") from exc

    if features.city not in info["cities"]:
        raise HTTPException(
            status_code=422,
            detail=f"Город '{features.city}' отсутствует в данных обучения. "
                   f"Допустимые варианты: {', '.join(info['cities'])}",
        )

    sample = pd.DataFrame([{
        "city": features.city,
        "total_square": features.total_square,
        "rooms": features.rooms,
        "floor": features.floor,
    }], columns=info["features"])

    try:
        raw_prediction = float(model.predict(sample)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ошибка при вычислении прогноза") from exc
    if not math.isfinite(raw_prediction):
        raise HTTPException(status_code=500, detail="Модель вернула некорректное значение")

    price = round(max(0.0, raw_prediction), 2)
    return PredictionResponse(
        predicted_price_rub=price,
        price_per_sqm_rub=round(price / features.total_square, 2),
        features=features,
    )


@app.get("/health", response_model=HealthResponse, tags=["Контроль состояния"], summary="Проверка доступности API и модели")
def health():
    """Liveness/health-check для демонстрации работоспособности сервиса."""
    try:
        load_resources()
    except Exception:
        raise HTTPException(status_code=503, detail="Модель не загрузилась")
    return HealthResponse(status="ok", model_loaded=True)


@app.get("/predict_get", response_model=PredictionResponse, tags=["Прогнозирование"], summary="Оценка квартиры через GET")
def predict_get(
    city: Annotated[str, Query(min_length=1, description="Город", examples=["Москва"])] = "Москва",
    total_square: Annotated[float, Query(ge=15, le=300, description="Общая площадь, м²")] = 60.0,
    rooms: Annotated[int, Query(ge=0, le=7, description="Число комнат; 0 — студия")] = 2,
    floor: Annotated[int, Query(ge=1, le=50, description="Этаж квартиры")] = 5,
):
    """Принимает значения через query-параметры адресной строки."""
    return make_prediction(ApartmentFeatures(
        city=city, total_square=total_square, rooms=rooms, floor=floor,
    ))


@app.post("/predict_post", response_model=PredictionResponse, tags=["Прогнозирование"], summary="Оценка квартиры через POST")
def predict_post(features: ApartmentFeatures):
    """Принимает признаки из JSON-тела запроса."""
    return make_prediction(features)


@app.get("/", tags=["Контроль состояния"], summary="Инструкция")
def index():
    return {
        "message": "API прогнозирования цены квартиры",
        "swagger": "/docs",
        "health": "/health",
        "get": "/predict_get",
        "post": "/predict_post",
    }
