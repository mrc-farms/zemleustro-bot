"""
collector.py — Async data collection module for ZemleustroBot.
Uses only free, no-personal-account APIs:
  - Open-Meteo Archive (climate)
  - Open-Meteo Elevation (DEM)
  - SoilGrids v2.0 ISRIC (soil)
  - Nominatim OSM (geocoding)
  - Overpass API (infrastructure)
  - PKK Rosreestr (cadastral)
"""

import asyncio
import datetime
import math
import logging
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two GPS points using Haversine formula."""
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_climate_date_range(years: int) -> tuple:
    """Return (start_date_str, end_date_str) covering N complete past calendar years."""
    end_year = datetime.date.today().year - 1
    start_year = end_year - years + 1
    return f"{start_year}-01-01", f"{end_year}-12-31"


# ---------------------------------------------------------------------------
# Climate — Open-Meteo Archive (ERA5)
# ---------------------------------------------------------------------------

async def fetch_climate(session: aiohttp.ClientSession, lat: float, lon: float, years: int = 1) -> Dict:
    """Fetch climate statistics from Open-Meteo Archive API for N complete past years."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    start_date, end_date = _get_climate_date_range(years)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
        "timezone": "auto",
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=45)) as resp:
            if resp.status != 200:
                return {"error": f"Open-Meteo HTTP {resp.status}"}
            data = await resp.json()

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])
        times = daily.get("time", [])

        if not temps or not times:
            return {"error": "Open-Meteo: пустые данные"}

        # Group by year for trend analysis
        year_temps: Dict[int, List[float]] = {}
        year_precip: Dict[int, float] = {}
        monthly_temps: List[List[float]] = [[] for _ in range(12)]
        monthly_precip: List[float] = [0.0] * 12

        for i, t in enumerate(times):
            try:
                yr = int(t[:4])
                month = int(t[5:7]) - 1
                if month < 0 or month > 11:
                    continue
                temp_val = temps[i] if i < len(temps) else None
                precip_val = precip[i] if i < len(precip) else None
                if temp_val is not None:
                    year_temps.setdefault(yr, []).append(temp_val)
                    monthly_temps[month].append(temp_val)
                if precip_val is not None:
                    year_precip[yr] = year_precip.get(yr, 0.0) + precip_val
                    monthly_precip[month] += precip_val
            except (IndexError, ValueError):
                continue

        # Per-year annual means for trend
        yearly_mean_temps = {yr: sum(v) / len(v) for yr, v in year_temps.items() if v}
        yearly_precip = {yr: round(yp, 1) for yr, yp in year_precip.items()}

        # Overall averages across all years
        all_temps = [v for vs in year_temps.values() for v in vs]
        all_precip_vals = list(year_precip.values())
        mean_temp = round(sum(all_temps) / len(all_temps), 1) if all_temps else None
        annual_precip = round(sum(all_precip_vals) / len(all_precip_vals), 1) if all_precip_vals else None

        temp_monthly = [
            round(sum(m) / len(m), 1) if m else None
            for m in monthly_temps
        ]
        veg_precip = sum(monthly_precip[4:9])
        veg_period_months = sum(1 for t in temp_monthly if t is not None and t > 5.0)

        # Trend: linear regression slope (°C/year) over available years
        temp_trend_c_per_year = None
        sorted_years = sorted(yearly_mean_temps.keys())
        if len(sorted_years) >= 2:
            n = len(sorted_years)
            xs = list(range(n))
            ys = [yearly_mean_temps[yr] for yr in sorted_years]
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            numerator = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
            denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))
            if denominator > 0:
                temp_trend_c_per_year = round(numerator / denominator, 3)

        return {
            "mean_temp_c": mean_temp,
            "annual_precip_mm": annual_precip,
            "veg_period_precip_mm": round(veg_precip, 1),
            "temp_monthly_c": temp_monthly,
            "veg_period_months": veg_period_months,
            "yearly_mean_temps": {str(yr): round(t, 1) for yr, t in yearly_mean_temps.items()},
            "yearly_precip_mm": {str(yr): p for yr, p in yearly_precip.items()},
            "temp_trend_c_per_year": temp_trend_c_per_year,
            "period": f"{start_date} — {end_date}",
            "years_analyzed": years,
            "source": "Open-Meteo Archive (ERA5)",
        }
    except asyncio.TimeoutError:
        return {"error": "Open-Meteo: таймаут запроса"}
    except aiohttp.ClientError as exc:
        return {"error": f"Open-Meteo: ошибка соединения — {exc}"}
    except Exception as exc:
        logger.exception("fetch_climate unexpected error")
        return {"error": f"Open-Meteo: непредвиденная ошибка — {exc}"}


# ---------------------------------------------------------------------------
# DEM — Open-Meteo Elevation (SRTM)
# ---------------------------------------------------------------------------

async def fetch_dem(session: aiohttp.ClientSession, lat: float, lon: float) -> Dict:
    """Fetch elevation and compute slope/aspect from SRTM via Open-Meteo."""
    url = "https://api.open-meteo.com/v1/elevation"
    lat_rad = math.radians(lat)
    dlat = 0.001
    dlon = 0.001 / math.cos(lat_rad) if math.cos(lat_rad) != 0 else 0.001

    # Center + 4 surrounding points: N, S, E, W
    points = [
        (lat, lon),
        (lat + dlat, lon),
        (lat - dlat, lon),
        (lat, lon + dlon),
        (lat, lon - dlon),
    ]
    lats_str = ",".join(str(p[0]) for p in points)
    lons_str = ",".join(str(p[1]) for p in points)

    try:
        async with session.get(
            url,
            params={"latitude": lats_str, "longitude": lons_str},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status != 200:
                return {"error": f"Open-Meteo Elevation HTTP {resp.status}"}
            data = await resp.json()

        elevations = data.get("elevation", [])
        if len(elevations) < 5:
            return {"error": "Open-Meteo Elevation: недостаточно данных"}

        elev_center = elevations[0]
        elev_n = elevations[1]
        elev_s = elevations[2]
        elev_e = elevations[3]
        elev_w = elevations[4]

        # Distance in metres for finite-difference denominator
        dist_ns = haversine(lat + dlat, lon, lat - dlat, lon)
        dist_ew = haversine(lat, lon + dlon, lat, lon - dlon)

        dz_dx = (elev_e - elev_w) / dist_ew if dist_ew > 0 else 0.0
        dz_dy = (elev_n - elev_s) / dist_ns if dist_ns > 0 else 0.0

        slope_rad = math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2))
        slope_deg = round(math.degrees(slope_rad), 2)

        # Aspect: 0=North, 90=East, 180=South, 270=West
        aspect_rad = math.atan2(dz_dx, dz_dy)
        aspect_deg = round((math.degrees(aspect_rad) + 360) % 360, 1)

        if 315 <= aspect_deg or aspect_deg < 45:
            aspect_text = "Северная экспозиция"
        elif 45 <= aspect_deg < 135:
            aspect_text = "Восточная экспозиция"
        elif 135 <= aspect_deg < 225:
            aspect_text = "Южная экспозиция"
        else:
            aspect_text = "Западная экспозиция"

        if slope_deg < 2:
            erosion_risk = "Низкий"
        elif slope_deg < 5:
            erosion_risk = "Средний"
        else:
            erosion_risk = "Высокий"

        return {
            "elevation_mean_m": round(elev_center, 1),
            "slope_deg": slope_deg,
            "aspect_deg": aspect_deg,
            "aspect_text": aspect_text,
            "erosion_risk": erosion_risk,
            "source": "SRTM via Open-Meteo",
        }
    except asyncio.TimeoutError:
        return {"error": "Open-Meteo Elevation: таймаут запроса"}
    except aiohttp.ClientError as exc:
        return {"error": f"Open-Meteo Elevation: ошибка соединения — {exc}"}
    except Exception as exc:
        logger.exception("fetch_dem unexpected error")
        return {"error": f"Open-Meteo Elevation: непредвиденная ошибка — {exc}"}


# ---------------------------------------------------------------------------
# Soil — SoilGrids v2.0 ISRIC
# ---------------------------------------------------------------------------

SOILGRIDS_SCALES = {
    "phh2o": 0.1,
    "soc": 0.1,
    "clay": 0.1,
    "bdod": 0.01,
    "cec": 0.1,
    "nitrogen": 0.01,
    "sand": 0.1,
    "silt": 0.1,
}

SOILGRIDS_UNITS = {
    "phh2o": "pH",
    "soc": "г/кг",
    "clay": "%",
    "bdod": "кг/дм³",
    "cec": "ммоль/кг",
    "nitrogen": "г/кг",
    "sand": "%",
    "silt": "%",
}


def _determine_soil_type(clay_pct: Optional[float], ph: Optional[float]) -> str:
    """Classify soil type from clay content and pH."""
    if clay_pct is None:
        if ph is not None:
            if ph < 5.5:
                return "Кислые почвы (данные по глине отсутствуют)"
            elif ph > 7.5:
                return "Щелочные почвы (данные по глине отсутствуют)"
        return "Не определён (нет данных)"

    if clay_pct < 15:
        soil = "Супесчаная/Лёгкосуглинистая почва"
    elif clay_pct < 25:
        soil = "Суглинистая почва"
    elif clay_pct < 40:
        soil = "Тяжелосуглинистая почва"
    else:
        soil = "Глинистая почва"

    if ph is not None:
        if ph < 5.5:
            soil += " (кислая)"
        elif ph < 6.5:
            soil += " (слабокислая)"
        elif ph <= 7.5:
            soil += " (нейтральная)"
        else:
            soil += " (щелочная)"

    return soil


async def _fetch_soilgrids_point(session: aiohttp.ClientSession, lat: float, lon: float) -> Optional[Dict]:
    """Fetch SoilGrids data for a single point. Returns raw averaged values dict or None."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    properties = ["phh2o", "soc", "clay", "bdod", "cec", "nitrogen", "sand", "silt"]
    depths = ["0-5cm", "5-15cm", "15-30cm"]

    params = {
        "lon": lon,
        "lat": lat,
        "property": properties,
        "depth": depths,
        "value": "mean",
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                logger.warning("SoilGrids HTTP %s for point (%.4f, %.4f)", resp.status, lat, lon)
                return None
            data = await resp.json()

        layers = data.get("properties", {}).get("layers", [])
        if not layers:
            return None

        result = {}
        for layer in layers:
            prop_name = layer.get("name", "")
            if prop_name not in SOILGRIDS_SCALES:
                continue
            scale = SOILGRIDS_SCALES[prop_name]
            depth_values = []
            for depth_info in layer.get("depths", []):
                val = depth_info.get("values", {}).get("mean")
                if val is not None:
                    depth_values.append(val * scale)
            if depth_values:
                result[prop_name] = sum(depth_values) / len(depth_values)

        return result if result else None
    except asyncio.TimeoutError:
        logger.warning("SoilGrids timeout for (%.4f, %.4f)", lat, lon)
        return None
    except Exception as exc:
        logger.warning("SoilGrids error for (%.4f, %.4f): %s", lat, lon, exc)
        return None


async def fetch_soilgrids(session: aiohttp.ClientSession, lat: float, lon: float) -> Dict:
    """Fetch and average SoilGrids data from 5 points within ~20m radius (cross pattern)."""
    try:
        lat_rad = math.radians(lat)
        # ~20 m offset: stays within any field, even narrow suburban plots
        dlat = 0.00018
        dlon = 0.00018 / math.cos(lat_rad) if math.cos(lat_rad) != 0 else 0.00018

        sample_points = [
            (lat, lon),
            (lat + dlat, lon),   # north  ~20 m
            (lat - dlat, lon),   # south  ~20 m
            (lat, lon + dlon),   # east   ~20 m
            (lat, lon - dlon),   # west   ~20 m
        ]

        point_results = []
        for i, (slat, slon) in enumerate(sample_points):
            result = await _fetch_soilgrids_point(session, slat, slon)
            point_results.append(result)
            if i < len(sample_points) - 1:
                await asyncio.sleep(0.4)  # Rate limit respect

        points = [p for p in point_results if p is not None]

        if not points:
            return {
                "data": {},
                "soil_type": "Не определён (нет данных SoilGrids)",
                "source": "SoilGrids v2.0 ISRIC",
                "points_sampled": 0,
            }

        # Average across available points
        all_keys: set = set()
        for p in points:
            all_keys.update(p.keys())

        averaged = {}
        for key in all_keys:
            vals = [p[key] for p in points if key in p]
            averaged[key] = sum(vals) / len(vals)

        soil_data = {}
        for prop, val in averaged.items():
            unit = SOILGRIDS_UNITS.get(prop, "")
            if prop == "phh2o":
                soil_data[prop] = {"value": round(val, 1), "unit": unit}
            elif prop in ("clay", "sand", "silt"):
                soil_data[prop] = {"value": round(val, 1), "unit": unit}
            else:
                soil_data[prop] = {"value": round(val, 2), "unit": unit}

        clay_pct = averaged.get("clay")
        ph = averaged.get("phh2o")
        soil_type = _determine_soil_type(clay_pct, ph)

        return {
            "data": soil_data,
            "soil_type": soil_type,
            "points_sampled": len(points),
            "source": "SoilGrids v2.0 ISRIC",
        }
    except Exception as exc:
        logger.exception("fetch_soilgrids unexpected error")
        return {
            "data": {},
            "soil_type": "Не определён (ошибка запроса)",
            "source": "SoilGrids v2.0 ISRIC",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Geocoding — Nominatim
# ---------------------------------------------------------------------------

async def fetch_geocoding(session: aiohttp.ClientSession, lat: float, lon: float) -> Dict:
    """Reverse geocode coordinates using Nominatim."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 12,
        "accept-language": "ru",
    }
    headers = {"User-Agent": "ZemleustroBot/1.0"}
    try:
        async with session.get(
            url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status != 200:
                return {"error": f"Nominatim HTTP {resp.status}"}
            data = await resp.json(content_type=None)

        address = data.get("address", {})
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
            or address.get("municipality")
            or "н/д"
        )
        return {
            "country": address.get("country", "н/д"),
            "state": address.get("state", "н/д"),
            "county": address.get("county", "н/д"),
            "city": city,
            "display_name": data.get("display_name", "н/д"),
        }
    except asyncio.TimeoutError:
        return {"error": "Nominatim: таймаут запроса"}
    except aiohttp.ClientError as exc:
        return {"error": f"Nominatim: ошибка соединения — {exc}"}
    except Exception as exc:
        logger.exception("fetch_geocoding unexpected error")
        return {"error": f"Nominatim: непредвиденная ошибка — {exc}"}


# ---------------------------------------------------------------------------
# OSM / Overpass API
# ---------------------------------------------------------------------------

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

HW_RU = {
    "motorway": "Автомагистраль",
    "trunk": "Федеральная трасса",
    "primary": "Дорога 1-й категории",
    "secondary": "Дорога 2-й категории",
    "tertiary": "Дорога 3-й категории",
    "unclassified": "Дорога местного значения",
    "residential": "Жилая дорога",
    "service": "Подъездная дорога",
    "track": "Грунтовая дорога",
    "path": "Тропа/просека",
    "footway": "Пешеходная дорожка",
    "road": "Дорога (неклассифицированная)",
}


async def overpass_query(session: aiohttp.ClientSession, query: str, timeout: int = 15) -> List[Dict]:
    """POST an Overpass QL query with failover across multiple servers."""
    data = {"data": query}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for server in OVERPASS_SERVERS:
        try:
            async with session.post(
                server,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    logger.warning("Overpass server %s returned HTTP %s", server, resp.status)
                    continue
                result = await resp.json(content_type=None)
                elements = result.get("elements", [])
                return elements
        except asyncio.TimeoutError:
            logger.warning("Overpass server %s timed out", server)
        except aiohttp.ClientError as exc:
            logger.warning("Overpass server %s connection error: %s", server, exc)
        except Exception as exc:
            logger.warning("Overpass server %s unexpected error: %s", server, exc)

    logger.error("All Overpass servers failed")
    return []


async def get_nearest_node(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    tag_filter: str,
    radius: int = 15000,
) -> Tuple[Optional[float], Dict]:
    """
    Find the nearest way matching tag_filter within radius metres.
    Returns (distance_metres, way_tags) or (None, {}).
    """
    query = (
        f"[out:json][timeout:13];"
        f"(way[{tag_filter}](around:{radius},{lat},{lon});>;);"
        f"out body;"
    )
    try:
        elements = await overpass_query(session, query, timeout=15)
        if not elements:
            return None, {}

        # Build node coordinate map
        nodes: Dict[int, Tuple[float, float]] = {}
        ways: List[Dict] = []

        for elem in elements:
            if elem.get("type") == "node":
                nid = elem.get("id")
                nlat = elem.get("lat")
                nlon = elem.get("lon")
                if nid is not None and nlat is not None and nlon is not None:
                    nodes[nid] = (nlat, nlon)
            elif elem.get("type") == "way":
                ways.append(elem)

        if not ways or not nodes:
            return None, {}

        min_dist = float("inf")
        best_tags: Dict = {}

        for way in ways:
            way_nodes = way.get("nodes", [])
            tags = way.get("tags", {})
            for nid in way_nodes:
                if nid in nodes:
                    nlat, nlon = nodes[nid]
                    dist = haversine(lat, lon, nlat, nlon)
                    if dist < min_dist:
                        min_dist = dist
                        best_tags = tags

        if min_dist == float("inf"):
            return None, {}

        return round(min_dist), best_tags
    except Exception as exc:
        logger.exception("get_nearest_node error for tag_filter=%s", tag_filter)
        return None, {}


async def fetch_osm_infrastructure(session: aiohttp.ClientSession, lat: float, lon: float) -> Dict:
    """Fetch OSM infrastructure data around the field in parallel."""
    try:
        # Run road, power, water, and pipeline queries in parallel
        road_task = get_nearest_node(session, lat, lon, '"highway"', radius=15000)
        power_task = get_nearest_node(session, lat, lon, '"power"="line"', radius=20000)
        water_task = get_nearest_node(session, lat, lon, '"waterway"~"river|stream|canal"', radius=15000)
        pipeline_task = get_nearest_node(session, lat, lon, '"man_made"="pipeline"]["substance"="gas"', radius=20000)

        settlements_query = (
            f"[out:json][timeout:13];"
            f'(node["place"~"city|town|village"](around:50000,{lat},{lon}););'
            f"out body;"
        )
        settlements_task = overpass_query(session, settlements_query, timeout=15)

        (road_dist, road_tags), (power_dist, _), (water_dist, _), (pipeline_dist, _), settlements = (
            await asyncio.gather(road_task, power_task, water_task, pipeline_task, settlements_task)
        )

        # Road type
        road_type = road_tags.get("highway", "unknown") if road_tags else "unknown"
        road_type_ru = HW_RU.get(road_type, road_type)
        truck_accessible = road_type in (
            "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential"
        )

        # Settlements: find nearest and top-3 distances
        nearest_settlement_m = None
        nearest_place_name = "н/д"
        route_distances_km: List[Dict] = []

        if settlements:
            settlement_list = []
            for s in settlements:
                slat = s.get("lat")
                slon = s.get("lon")
                stags = s.get("tags", {})
                sname = stags.get("name") or stags.get("name:ru") or "Населённый пункт"
                if slat is not None and slon is not None:
                    dist = haversine(lat, lon, slat, slon)
                    settlement_list.append((dist, sname))

            if settlement_list:
                settlement_list.sort(key=lambda x: x[0])
                nearest_settlement_m = round(settlement_list[0][0])
                nearest_place_name = settlement_list[0][1]
                for dist, name in settlement_list[:3]:
                    route_distances_km.append({"name": name, "distance_km": round(dist / 1000, 1)})

        return {
            "nearest_road_m": road_dist,
            "road_type": road_type,
            "road_type_ru": road_type_ru,
            "truck_accessible": truck_accessible,
            "nearest_powerline_m": power_dist,
            "nearest_waterway_m": water_dist,
            "nearest_gas_pipeline_m": pipeline_dist,
            "nearest_settlement_m": nearest_settlement_m,
            "nearest_place_name": nearest_place_name,
            "route_distances_km": route_distances_km,
            "source": "OpenStreetMap via Overpass API",
        }
    except Exception as exc:
        logger.exception("fetch_osm_infrastructure unexpected error")
        return {"error": f"OSM инфраструктура: непредвиденная ошибка — {exc}"}


# ---------------------------------------------------------------------------
# Cadastral — PKK Rosreestr
# ---------------------------------------------------------------------------

LAND_CATEGORIES = {
    "003001000000": "Земли сельскохозяйственного назначения",
    "003002000000": "Земли населённых пунктов",
    "003003000000": "Земли промышленности и иного специального назначения",
    "003004000000": "Земли особо охраняемых территорий",
    "003005000000": "Земли лесного фонда",
    "003006000000": "Земли водного фонда",
    "003007000000": "Земли запаса",
    "0": "Не установлена",
}

OWN_TYPES = {
    1: "Частная",
    2: "Государственная",
    3: "Муниципальная",
}


async def fetch_rosreestr(session: aiohttp.ClientSession, lat: float, lon: float) -> Dict:
    """Fetch cadastral information from PKK Rosreestr, including address and cadastral value."""
    search_url = (
        f"https://pkk.rosreestr.ru/api/features/1"
        f"?text={lat}+{lon}&tolerance=4&returnGeometry=false&limit=5"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://pkk.rosreestr.ru/",
        "Accept": "application/json, text/plain, */*",
    }
    try:
        async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return {
                    "error": f"PKK Rosreestr HTTP {resp.status}",
                    "source": "ПКК Росреестр",
                }
            data = await resp.json(content_type=None)

        features = data.get("features", []) or []
        if not features:
            return {
                "error": (
                    "Участок не найден в ПКК Росреестр. "
                    "Возможно, координаты находятся вне кадастровой карты или участок не размежёван."
                ),
                "source": "ПКК Росреестр",
            }

        feature = features[0]
        attrs = feature.get("attrs", {}) or {}

        cad_num = attrs.get("cn") or attrs.get("id") or "н/д"
        area_m2 = attrs.get("area_value")
        category_code = str(attrs.get("category_type", "0"))
        category = LAND_CATEGORIES.get(category_code, f"Код: {category_code}")
        permitted_use = attrs.get("util_by_doc") or attrs.get("util_code_doc") or "н/д"
        own_code = attrs.get("own_type")
        ownership_type = OWN_TYPES.get(own_code, "н/д") if own_code is not None else "н/д"

        # Fetch detail endpoint for address and cadastral value
        address = "н/д"
        cadastral_value_rub = None
        cadastral_value_date = None

        if cad_num and cad_num != "н/д":
            encoded_id = cad_num.replace(":", "%3A")
            detail_url = f"https://pkk.rosreestr.ru/api/features/1/{encoded_id}"
            try:
                async with session.get(
                    detail_url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)
                ) as detail_resp:
                    if detail_resp.status == 200:
                        detail_data = await detail_resp.json(content_type=None)
                        detail_attrs = detail_data.get("feature", {}).get("attrs", {}) or {}
                        address = detail_attrs.get("address") or "н/д"
                        cad_cost = detail_attrs.get("cad_cost")
                        if cad_cost is not None:
                            try:
                                cadastral_value_rub = float(cad_cost)
                            except (TypeError, ValueError):
                                pass
                        cadastral_value_date = detail_attrs.get("date_cost") or None
            except Exception as exc:
                logger.warning("PKK detail endpoint failed for %s: %s", cad_num, exc)

        return {
            "cadastral_number": cad_num,
            "address": address,
            "area_m2": area_m2,
            "category": category,
            "permitted_use": permitted_use,
            "ownership_type": ownership_type,
            "cadastral_value_rub": cadastral_value_rub,
            "cadastral_value_date": cadastral_value_date,
            "source": "ПКК Росреестр",
        }
    except asyncio.TimeoutError:
        return {"error": "ПКК Росреестр: таймаут запроса", "source": "ПКК Росреестр"}
    except aiohttp.ClientError as exc:
        return {"error": f"ПКК Росреестр: ошибка соединения — {exc}", "source": "ПКК Росреестр"}
    except Exception as exc:
        logger.exception("fetch_rosreestr unexpected error")
        return {"error": f"ПКК Росреестр: непредвиденная ошибка — {exc}", "source": "ПКК Росреестр"}


# ---------------------------------------------------------------------------
# Main collection functions
# ---------------------------------------------------------------------------

async def collect_field_data(
    lat: float, lon: float, field_name: str = "Участок", years: int = 1
) -> Dict:
    """
    Collect all data for a single field.
    Creates one shared aiohttp.ClientSession and runs all requests in parallel.
    years: number of complete past calendar years to analyze (1, 3, or 5).
    """
    start_date, end_date = _get_climate_date_range(years)
    connector = aiohttp.TCPConnector(limit=10, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_climate(session, lat, lon, years),
            fetch_dem(session, lat, lon),
            fetch_soilgrids(session, lat, lon),
            fetch_geocoding(session, lat, lon),
            fetch_osm_infrastructure(session, lat, lon),
            fetch_rosreestr(session, lat, lon),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    def safe_result(r, name: str) -> Dict:
        if isinstance(r, Exception):
            logger.error("Task %s raised exception: %s", name, r)
            return {"error": str(r)}
        return r

    names = ["climate", "dem", "soilgrids", "geo", "osm", "rosreestr"]
    climate, dem, soilgrids, geo, osm, rosreestr = [
        safe_result(r, n) for r, n in zip(results, names)
    ]

    return {
        "meta": {
            "lat": lat,
            "lon": lon,
            "name": field_name,
            "years": years,
            "period": f"{start_date} — {end_date}",
        },
        "raw": {
            "climate": climate,
            "dem": dem,
            "soilgrids": soilgrids,
            "geo": geo,
            "osm": osm,
            "rosreestr": rosreestr,
        },
    }


async def collect_multiple_fields(fields_list: List[Dict], years: int = 1) -> Dict:
    """
    Collect data for multiple fields sequentially to avoid rate limits.
    fields_list: [{"lat": ..., "lon": ..., "name": ...}, ...]
    years: number of complete past calendar years to analyze (1, 3, or 5).
    Returns: {"Field_1": field_data, "Field_2": field_data, ...}
    """
    start_date, end_date = _get_climate_date_range(years)
    results = {}
    for i, field in enumerate(fields_list):
        field_id = f"Field_{i + 1}"
        lat = field.get("lat", 0.0)
        lon = field.get("lon", 0.0)
        name = field.get("name", f"Участок {i + 1}")
        logger.info("Collecting data for %s (%s, %s)...", name, lat, lon)
        try:
            data = await collect_field_data(lat, lon, name, years)
        except Exception as exc:
            logger.exception("collect_field_data failed for field %s", field_id)
            data = {
                "meta": {"lat": lat, "lon": lon, "name": name, "years": years, "period": f"{start_date} — {end_date}"},
                "raw": {"error": str(exc)},
            }
        results[field_id] = data
        # Small delay between fields to be polite to external APIs
        if i < len(fields_list) - 1:
            await asyncio.sleep(1.0)

    return results
