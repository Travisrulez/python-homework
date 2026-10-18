"""Интерфейс прогнозирования цены квартиры.
Запуск: streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title='Оценка квартиры · HW 9',
    page_icon='🏠',
    layout='centered',
)

st.markdown('''
<style>
.main .block-container {max-width: 850px; padding-top: 2.3rem;}
[data-testid="stForm"] {border: 1px solid rgba(128,128,128,.22); border-radius: 18px; padding: 1.1rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.20); border-radius: 14px; padding: 0.75rem 1rem;}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = joblib.load(BASE / 'model.joblib')
    info = json.loads((BASE / 'model_info.json').read_text(encoding='utf-8'))
    return model, info


def rub(value: float) -> str:
    return f'{value:,.0f}'.replace(',', ' ') + ' ₽'


st.title('🏠 Оценка стоимости квартиры')
st.write('Учебное приложение: предсказание цены по площади, комнатности, этажу и городу на основе объявлений Москвы и Подмосковья.')

try:
    model, info = load_resources()
except Exception as exc:
    st.error('Файлы обученной модели не найдены или повреждены. Сначала выполните команду `python train_model.py`.')
    st.exception(exc)
    st.stop()

with st.form('predict_form'):
    st.subheader('Параметры квартиры')
    cities = info['cities']
    selected_city = st.selectbox('Город', cities, index=cities.index('Москва') if 'Москва' in cities else 0)
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input('Общая площадь, м²', min_value=15.0, max_value=300.0, value=55.0, step=1.0)
        rooms = st.selectbox('Количество комнат', options=list(range(0, 8)), index=2,
                              format_func=lambda x: 'Студия' if x == 0 else str(x))
    with c2:
        floor = st.number_input('Этаж', min_value=1, max_value=50, value=7, step=1)
        st.caption('Диапазоны ввода соответствуют данным, используемым при обучении.')
    submit = st.form_submit_button('Рассчитать стоимость', type='primary', use_container_width=True)

if submit:
    features = pd.DataFrame([{
        'city': selected_city,
        'total_square': float(area),
        'rooms': int(rooms),
        'floor': int(floor),
    }])
    try:
        predicted_price = max(0.0, float(model.predict(features)[0]))
    except Exception as exc:
        st.error('Не удалось выполнить прогноз. Проверьте значения признаков и совместимость библиотек.')
        st.exception(exc)
    else:
        st.success('Прогноз успешно рассчитан')
        st.metric('Прогнозируемая стоимость', rub(predicted_price))
        st.caption(f'Ориентировочная стоимость за м²: {rub(predicted_price / area)}')
        st.info('Прогноз учебный: модель обучена на ценах объявлений, а не на фактических сделках. Не является оценочным заключением.')

st.divider()
st.subheader('О модели')
a, b, c = st.columns(3)
a.metric('Объявлений в обучающей выборке', f"{info['n_train']:,}".replace(',', ' '))
b.metric('MAE на тесте', rub(info['mae_rub']))
c.metric('R² на тесте', f"{info['r2']:.2f}")
st.caption('Использована HistGradientBoostingRegressor, метрика вычислена на отложенной тестовой выборке (20%, random_state=42).')
