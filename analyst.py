"""
analyst.py — Groq-based AI agronomic analysis module for ZemleustroBot.
Uses llama-3.3-70b-versatile via Groq API to generate field reports.
"""

import os
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

EXPERT_SYSTEM_PROMPT = """\
ТЫ — ЭКСПЕРТ ПО СОЗДАНИЮ СЕЛЬСКОХОЗЯЙСТВЕННЫХ ПРЕДПРИЯТИЙ И СПЕЦИАЛИСТ ПО ДИСТАНЦИОННОМУ ЗОНДИРОВАНИЮ ЗЕМЛИ.
Твоя роль: Главный агроном + почвовед + аналитик. Ты анализируешь реальные данные и формируешь профессиональные агрономические выводы.
При анализе данных:
- Интерпретируешь pH почвы (оптимум 6.0-7.0 для большинства культур)
- Оцениваешь SOC: <10 г/кг - бедная, 10-20 - средняя, >20 - богатая почва
- Анализируешь риск эрозии по уклону и экспозиции
- Оцениваешь логистику по расстоянию до дорог и населённых пунктов
- Интерпретируешь климатические данные агрономически
- Даёшь конкретные агрономические рекомендации
Пишешь только на русском языке. Используешь ТОЛЬКО предоставленные данные. Не выдумываешь цифры.\
"""


def ask_expert(prompt: str, api_key: str, max_tokens: int = 2000) -> str:
    """
    Send a prompt to Groq llama-3.3-70b-versatile and return the response text.
    Tries up to 2 times with a 15-second pause between attempts.
    Returns the model's text response, or an error string on failure.
    """
    try:
        from groq import Groq
    except ImportError:
        return "Ошибка: библиотека groq не установлена. Выполните: pip install groq"

    if not api_key:
        return "Ошибка: GROQ_API_KEY не задан."

    client = Groq(api_key=api_key)

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = completion.choices[0].message.content
            if text:
                return text.strip()
            return "Модель вернула пустой ответ."
        except Exception as exc:
            logger.warning("ask_expert attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(15)

    return f"Ошибка: не удалось получить ответ от Groq после 2 попыток."


def _fmt(value, unit: str = "", fallback: str = "н/д") -> str:
    """Format a value with unit, or return fallback if value is None/missing."""
    if value is None:
        return fallback
    if unit:
        return f"{value} {unit}"
    return str(value)


def _get_soil_line(soilgrids: Dict, prop: str) -> str:
    """Extract a formatted soil property value from soilgrids dict."""
    data = soilgrids.get("data", {})
    entry = data.get(prop, {})
    val = entry.get("value")
    unit = entry.get("unit", "")
    if val is None:
        return "н/д"
    return f"{val} {unit}".strip()


def generate_field_report(field_data: Dict, api_key: str) -> str:
    """
    Build a comprehensive Russian agronomic report for a single field.
    Extracts all available data, builds a prompt, and calls ask_expert.
    Returns the full report text.
    """
    meta = field_data.get("meta", {})
    raw = field_data.get("raw", {})

    lat = meta.get("lat", "?")
    lon = meta.get("lon", "?")
    name = meta.get("name", "Участок")
    years = meta.get("years", 1)
    period = meta.get("period", "н/д")

    climate = raw.get("climate", {}) if isinstance(raw.get("climate"), dict) else {}
    dem = raw.get("dem", {}) if isinstance(raw.get("dem"), dict) else {}
    soilgrids = raw.get("soilgrids", {}) if isinstance(raw.get("soilgrids"), dict) else {}
    geo = raw.get("geo", {}) if isinstance(raw.get("geo"), dict) else {}
    osm = raw.get("osm", {}) if isinstance(raw.get("osm"), dict) else {}
    rosreestr = raw.get("rosreestr", {}) if isinstance(raw.get("rosreestr"), dict) else {}

    # --- Location ---
    country = geo.get("country", "н/д")
    state = geo.get("state", "н/д")
    county = geo.get("county", "н/д")
    city = geo.get("city", "н/д")
    display_name = geo.get("display_name", "н/д")

    # --- Climate ---
    mean_temp = _fmt(climate.get("mean_temp_c"), "°C")
    annual_precip = _fmt(climate.get("annual_precip_mm"), "мм")
    veg_precip = _fmt(climate.get("veg_period_precip_mm"), "мм")
    veg_months = _fmt(climate.get("veg_period_months"), "мес.")
    temp_monthly = climate.get("temp_monthly_c", [])
    climate_period = climate.get("period", period)
    climate_error = climate.get("error", "")
    temp_trend = climate.get("temp_trend_c_per_year")
    yearly_mean_temps = climate.get("yearly_mean_temps", {})
    yearly_precip = climate.get("yearly_precip_mm", {})

    monthly_str = "н/д"
    if temp_monthly and len(temp_monthly) == 12:
        month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                       "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
        parts = []
        for mn, tv in zip(month_names, temp_monthly):
            parts.append(f"{mn}: {tv if tv is not None else 'н/д'}°C")
        monthly_str = ", ".join(parts)

    trend_str = "н/д"
    if temp_trend is not None:
        direction = "потепление" if temp_trend > 0 else "похолодание"
        trend_str = f"{'+' if temp_trend > 0 else ''}{temp_trend} °C/год ({direction})"

    yearly_temp_str = "н/д"
    if yearly_mean_temps:
        yearly_temp_str = ", ".join(
            f"{yr}: {t}°C" for yr, t in sorted(yearly_mean_temps.items())
        )

    yearly_precip_str = "н/д"
    if yearly_precip:
        yearly_precip_str = ", ".join(
            f"{yr}: {p} мм" for yr, p in sorted(yearly_precip.items())
        )

    # --- DEM ---
    elevation = _fmt(dem.get("elevation_mean_m"), "м")
    slope = _fmt(dem.get("slope_deg"), "°")
    aspect_text = dem.get("aspect_text", "н/д")
    erosion_risk = dem.get("erosion_risk", "н/д")
    dem_error = dem.get("error", "")

    # --- Soil ---
    soil_type = soilgrids.get("soil_type", "н/д")
    soil_points = soilgrids.get("points_sampled", "н/д")
    ph_val = _get_soil_line(soilgrids, "phh2o")
    soc_val = _get_soil_line(soilgrids, "soc")
    clay_val = _get_soil_line(soilgrids, "clay")
    sand_val = _get_soil_line(soilgrids, "sand")
    silt_val = _get_soil_line(soilgrids, "silt")
    bdod_val = _get_soil_line(soilgrids, "bdod")
    cec_val = _get_soil_line(soilgrids, "cec")
    nitrogen_val = _get_soil_line(soilgrids, "nitrogen")
    soil_error = soilgrids.get("error", "")

    # --- Infrastructure (OSM) ---
    road_dist = _fmt(osm.get("nearest_road_m"), "м")
    road_type_ru = osm.get("road_type_ru", "н/д")
    truck_accessible = osm.get("truck_accessible")
    truck_str = "Да" if truck_accessible else ("Нет" if truck_accessible is not None else "н/д")
    powerline_dist = _fmt(osm.get("nearest_powerline_m"), "м")
    waterway_dist = _fmt(osm.get("nearest_waterway_m"), "м")
    pipeline_dist = _fmt(osm.get("nearest_gas_pipeline_m"), "м")
    settlement_dist = _fmt(osm.get("nearest_settlement_m"), "м")
    place_name = osm.get("nearest_place_name", "н/д")
    route_distances = osm.get("route_distances_km", [])
    osm_error = osm.get("error", "")

    routes_str = "н/д"
    if route_distances:
        routes_str = "; ".join(
            f"{r.get('name', '?')} — {r.get('distance_km', '?')} км"
            for r in route_distances
        )

    # --- Cadastral ---
    cad_num = rosreestr.get("cadastral_number", "н/д")
    address = rosreestr.get("address", "н/д")
    area_m2 = rosreestr.get("area_m2")
    area_str = f"{area_m2} м²" if area_m2 else "н/д"
    category = rosreestr.get("category", "н/д")
    permitted_use = rosreestr.get("permitted_use", "н/д")
    ownership = rosreestr.get("ownership_type", "н/д")
    cad_value = rosreestr.get("cadastral_value_rub")
    cad_value_date = rosreestr.get("cadastral_value_date", "н/д")
    cad_value_str = f"{cad_value:,.0f} руб." if cad_value else "н/д"
    rosreestr_error = rosreestr.get("error", "")

    # Build prompt
    lines = [
        f"АГРОНОМИЧЕСКИЙ АНАЛИЗ УЧАСТКА: {name}",
        f"Координаты: {lat}, {lon} | Период анализа: {period} ({years} лет)",
        "",
        "═══ МЕСТОПОЛОЖЕНИЕ ═══",
        f"Страна: {country}",
        f"Регион (субъект): {state}",
        f"Район: {county}",
        f"Ближайший населённый пункт: {city}",
        f"Полный адрес: {display_name}",
        "",
        "═══ РЕЛЬЕФ И ВЫСОТА ═══",
    ]

    if dem_error:
        lines.append(f"Данные рельефа недоступны: {dem_error}")
    else:
        lines += [
            f"Высота над уровнем моря: {elevation}",
            f"Уклон: {slope}",
            f"Экспозиция: {aspect_text}",
            f"Риск эрозии: {erosion_risk}",
        ]

    lines += [
        "",
        "═══ КЛИМАТ ═══",
    ]
    if climate_error:
        lines.append(f"Данные климата недоступны: {climate_error}")
    else:
        lines += [
            f"Период: {climate_period}",
            f"Среднегодовая температура (среднее за период): {mean_temp}",
            f"Годовое количество осадков (среднее): {annual_precip}",
            f"Осадки за вегетационный период (май–сентябрь): {veg_precip}",
            f"Вегетационный период (месяцев >5°C): {veg_months}",
            f"Среднемесячные температуры (среднее за период): {monthly_str}",
        ]
        if years > 1:
            lines += [
                f"Температура по годам: {yearly_temp_str}",
                f"Осадки по годам: {yearly_precip_str}",
                f"Тренд температуры: {trend_str}",
            ]

    lines += [
        "",
        "═══ ПОЧВЫ (слои 0–30 см) ═══",
    ]
    if soil_error and not soilgrids.get("data"):
        lines.append(f"Данные почвы недоступны: {soil_error}")
    else:
        lines += [
            f"Тип почвы: {soil_type}",
            f"Точек отбора проб: {soil_points} (крестообразная сетка ~500 м)",
            f"pH (вода): {ph_val}",
            f"Органический углерод (SOC): {soc_val}",
            f"Содержание глины: {clay_val}",
            f"Содержание песка: {sand_val}",
            f"Содержание ила: {silt_val}",
            f"Объёмная плотность: {bdod_val}",
            f"ЕКО (CEC): {cec_val}",
            f"Общий азот: {nitrogen_val}",
        ]

    lines += [
        "",
        "═══ ИНФРАСТРУКТУРА (OSM) ═══",
    ]
    if osm_error:
        lines.append(f"Данные инфраструктуры недоступны: {osm_error}")
    else:
        lines += [
            f"Расстояние до ближайшей дороги: {road_dist} ({road_type_ru})",
            f"Доступность для грузового транспорта: {truck_str}",
            f"Расстояние до ЛЭП: {powerline_dist}",
            f"Расстояние до водотока: {waterway_dist}",
            f"Расстояние до газопровода: {pipeline_dist}",
            f"Ближайший населённый пункт: {place_name} — {settlement_dist}",
            f"Ближайшие населённые пункты (топ-3): {routes_str}",
        ]

    lines += [
        "",
        "═══ КАДАСТРОВЫЕ ДАННЫЕ ═══",
    ]
    if rosreestr_error:
        lines.append(f"Кадастровые данные: {rosreestr_error}")
    else:
        lines += [
            f"Кадастровый номер: {cad_num}",
            f"Адрес по кадастру: {address}",
            f"Площадь: {area_str}",
            f"Категория земель: {category}",
            f"Разрешённое использование: {permitted_use}",
            f"Форма собственности: {ownership}",
            f"Кадастровая стоимость: {cad_value_str} (дата оценки: {cad_value_date})",
        ]

    lines += [
        "",
        "═══ ЗАДАНИЕ ═══",
        "На основе ТОЛЬКО предоставленных выше данных составь профессиональный агрономический отчёт.",
        "Структура отчёта — строго 7 разделов:",
        "1. ПОЛОЖЕНИЕ И РЕЛЬЕФ — оценка местоположения, высоты, уклона, экспозиции и рисков",
        "2. ПОЧВЫ — детальная характеристика почвы, плодородие, лимитирующие факторы",
        "3. КЛИМАТ — агроклиматическая оценка; если данные за несколько лет — опиши тренды",
        "4. ВОДНЫЙ РЕЖИМ — оценка водного питания, ирригационная потребность",
        "5. ЭКОЛОГИЯ — риски эрозии, деградации, охранные зоны",
        "6. ЛОГИСТИКА — оценка транспортной доступности, близость к рынкам, кадастровые данные",
        "7. РЕКОМЕНДАЦИИ — конкретные культуры, агротехнические меры, приоритеты",
        "Используй только данные из отчёта. Не придумывай цифры. Пиши профессионально и по-русски.",
    ]

    prompt = "\n".join(lines)
    return ask_expert(prompt, api_key, max_tokens=3000)


def generate_region_summary(all_fields_data: Dict, api_key: str) -> str:
    """
    Generate a regional summary based on all collected field data.
    Returns the summary text.
    """
    lines = [
        "СВОДКА ПО РЕГИОНУ",
        f"Количество обследованных участков: {len(all_fields_data)}",
        "",
    ]

    for field_id, field_data in all_fields_data.items():
        meta = field_data.get("meta", {})
        raw = field_data.get("raw", {})
        name = meta.get("name", field_id)
        lat = meta.get("lat", "?")
        lon = meta.get("lon", "?")

        geo = raw.get("geo", {}) if isinstance(raw.get("geo"), dict) else {}
        climate = raw.get("climate", {}) if isinstance(raw.get("climate"), dict) else {}
        soilgrids = raw.get("soilgrids", {}) if isinstance(raw.get("soilgrids"), dict) else {}

        state = geo.get("state", "н/д")
        mean_temp = _fmt(climate.get("mean_temp_c"), "°C")
        annual_precip = _fmt(climate.get("annual_precip_mm"), "мм")
        soil_type = soilgrids.get("soil_type", "н/д")
        ph_val = _get_soil_line(soilgrids, "phh2o")
        soc_val = _get_soil_line(soilgrids, "soc")

        lines += [
            f"--- {name} ({lat}, {lon}) ---",
            f"  Регион: {state}",
            f"  Среднегодовая температура: {mean_temp}",
            f"  Годовые осадки: {annual_precip}",
            f"  Тип почвы: {soil_type}",
            f"  pH: {ph_val}",
            f"  SOC: {soc_val}",
            "",
        ]

    lines += [
        "═══ ЗАДАНИЕ ═══",
        "На основе данных по всем участкам составь краткую сводку по региону:",
        "- Общая агроклиматическая характеристика региона",
        "- Общие почвенные условия",
        "- Основные агропроизводственные возможности региона",
        "- Региональные риски и ограничения",
        "Пиши кратко, по-русски, только на основе предоставленных данных.",
    ]

    prompt = "\n".join(lines)
    return ask_expert(prompt, api_key, max_tokens=2000)


def generate_conclusion(all_fields_data: Dict, api_key: str) -> str:
    """
    Generate a comparative conclusion across all fields with recommendations.
    Returns the conclusion text.
    """
    lines = [
        "СРАВНИТЕЛЬНЫЙ АНАЛИЗ УЧАСТКОВ",
        f"Всего участков: {len(all_fields_data)}",
        "",
    ]

    for field_id, field_data in all_fields_data.items():
        meta = field_data.get("meta", {})
        raw = field_data.get("raw", {})
        name = meta.get("name", field_id)
        lat = meta.get("lat", "?")
        lon = meta.get("lon", "?")

        geo = raw.get("geo", {}) if isinstance(raw.get("geo"), dict) else {}
        climate = raw.get("climate", {}) if isinstance(raw.get("climate"), dict) else {}
        dem = raw.get("dem", {}) if isinstance(raw.get("dem"), dict) else {}
        soilgrids = raw.get("soilgrids", {}) if isinstance(raw.get("soilgrids"), dict) else {}
        osm = raw.get("osm", {}) if isinstance(raw.get("osm"), dict) else {}
        rosreestr = raw.get("rosreestr", {}) if isinstance(raw.get("rosreestr"), dict) else {}

        state = geo.get("state", "н/д")
        mean_temp = _fmt(climate.get("mean_temp_c"), "°C")
        annual_precip = _fmt(climate.get("annual_precip_mm"), "мм")
        veg_months = _fmt(climate.get("veg_period_months"), "мес.")
        elevation = _fmt(dem.get("elevation_mean_m"), "м")
        slope = _fmt(dem.get("slope_deg"), "°")
        erosion_risk = dem.get("erosion_risk", "н/д")
        soil_type = soilgrids.get("soil_type", "н/д")
        ph_val = _get_soil_line(soilgrids, "phh2o")
        soc_val = _get_soil_line(soilgrids, "soc")
        clay_val = _get_soil_line(soilgrids, "clay")
        road_dist = _fmt(osm.get("nearest_road_m"), "м")
        road_type_ru = osm.get("road_type_ru", "н/д")
        truck_accessible = osm.get("truck_accessible")
        truck_str = "Да" if truck_accessible else ("Нет" if truck_accessible is not None else "н/д")
        settlement_dist = _fmt(osm.get("nearest_settlement_m"), "м")
        category = rosreestr.get("category", "н/д")
        area_m2 = rosreestr.get("area_m2")
        area_str = f"{area_m2} м²" if area_m2 else "н/д"

        lines += [
            f"=== {name} ({lat}, {lon}) ===",
            f"  Регион: {state}",
            f"  Климат: t={mean_temp}, осадки={annual_precip}, вег. период={veg_months}",
            f"  Рельеф: высота={elevation}, уклон={slope}, риск эрозии={erosion_risk}",
            f"  Почва: {soil_type} | pH={ph_val} | SOC={soc_val} | Глина={clay_val}",
            f"  Дорога: {road_dist} ({road_type_ru}) | Грузовик: {truck_str}",
            f"  До нас. пункта: {settlement_dist}",
            f"  Категория земель: {category}",
            f"  Площадь: {area_str}",
            "",
        ]

    lines += [
        "═══ ЗАДАНИЕ ═══",
        "На основе данных по всем участкам проведи сравнительный анализ и дай итоговые рекомендации:",
        "1. СРАВНЕНИЕ УЧАСТКОВ — сравнительная таблица ключевых показателей",
        "2. ЛУЧШИЙ УЧАСТОК — какой участок наиболее перспективен и почему",
        "3. РЕКОМЕНДУЕМЫЕ КУЛЬТУРЫ — для каждого участка, с обоснованием",
        "4. АГРОТЕХНИЧЕСКИЕ МЕРЫ — приоритетные мероприятия для улучшения каждого участка",
        "5. ИНВЕСТИЦИОННЫЕ ПРИОРИТЕТЫ — в порядке убывания приоритетности",
        "Используй только предоставленные данные. Пиши профессионально и по-русски.",
    ]

    prompt = "\n".join(lines)
    return ask_expert(prompt, api_key, max_tokens=2500)
