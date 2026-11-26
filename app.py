from pathlib import Path
import math
import json
import time
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="India Shop Business Advisor", page_icon="🏪", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "shop_demand_model.joblib"
RAW_DIR = BASE_DIR / "Data" / "Raw"
CACHE_V2 = RAW_DIR / "osm_places_cache_v2.csv"
OSM_CACHE_PATHS = [
    CACHE_V2,
    RAW_DIR / "osm_places_cache.csv",
    BASE_DIR / "Data" / "processed" / "osm_places_cache.csv",
    BASE_DIR / "Data" / "processed" / "kochi_businesses_cleaned.csv",
    RAW_DIR / "kochi_businesses_osm.csv",
]
HEADERS = {"User-Agent": "IndiaShopBusinessAdvisor/1.0 (educational project)"}
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

# Fallbacks are specific configured points only, not universal locality-centroids.
LOCATION_FALLBACKS = {
    ("kakkanad", "kochi", "kerala"): {
        "lat": 10.01636, "lon": 76.34168,
        "display_name": "Kakkanad, Kochi, Kerala (configured fallback point)",
        "place_type": "configured_fallback", "place_class": "fallback"
    }
}


PROFILES = {
    "Grocery": {"avg_bill": 300, "cogs": .78, "staff": 22000, "utilities": 9000, "marketing": 5000, "setup": 140000, "stock": 220000, "base": 600},
    "Cafe": {"avg_bill": 180, "cogs": .35, "staff": 30000, "utilities": 12000, "marketing": 7000, "setup": 220000, "stock": 80000, "base": 450},
    "Stationery": {"avg_bill": 250, "cogs": .55, "staff": 18000, "utilities": 6000, "marketing": 3000, "setup": 100000, "stock": 150000, "base": 350},
    "Laundry": {"avg_bill": 300, "cogs": .20, "staff": 20000, "utilities": 14000, "marketing": 4000, "setup": 180000, "stock": 40000, "base": 220},
    "Computer Repair": {"avg_bill": 500, "cogs": .15, "staff": 22000, "utilities": 7000, "marketing": 5000, "setup": 130000, "stock": 70000, "base": 120},
    "Pharmacy": {"avg_bill": 350, "cogs": .72, "staff": 26000, "utilities": 8000, "marketing": 4000, "setup": 180000, "stock": 300000, "base": 400},
    "Bakery": {"avg_bill": 160, "cogs": .38, "staff": 24000, "utilities": 14000, "marketing": 5000, "setup": 200000, "stock": 70000, "base": 300},
    "Clothing": {"avg_bill": 800, "cogs": .55, "staff": 22000, "utilities": 8000, "marketing": 6000, "setup": 150000, "stock": 250000, "base": 350},
    "Restaurant": {"avg_bill": 250, "cogs": .38, "staff": 38000, "utilities": 18000, "marketing": 8000, "setup": 350000, "stock": 120000, "base": 500},
}


def money(value):
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "Unavailable"


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, str(exc)


def fallback_location(locality, city, state):
    return LOCATION_FALLBACKS.get((locality.strip().lower(), city.strip().lower(), state.strip().lower()))


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(locality, city, state):
    """Resolve locality using Photon first, then Nominatim; city fallback is clearly labeled."""
    locality = (locality or "").strip()
    city = (city or "").strip()
    state = (state or "").strip()
    if not locality or not city or not state:
        return None

    # Photon is the primary geocoder. It is often more tolerant of locality-name variations.
    photon_queries = list(dict.fromkeys([
        f"{locality}, {city}, {state}, India",
        f"{locality}, {state}, India",
        f"{city}, {state}, India",
    ]))
    for query_index, query in enumerate(photon_queries):
        try:
            response = requests.get("https://photon.komoot.io/api/", params={
                "q": query, "limit": 8, "lang": "en"
            }, headers=HEADERS, timeout=(3, 8))
            response.raise_for_status()
            features = response.json().get("features", [])
            ranked = []
            for feature in features:
                props = feature.get("properties", {}) or {}
                coords = (feature.get("geometry", {}) or {}).get("coordinates", [])
                if len(coords) < 2:
                    continue
                name = str(props.get("name", ""))
                city_prop = str(props.get("city", props.get("county", props.get("district", ""))))
                state_prop = str(props.get("state", ""))
                country_prop = str(props.get("country", ""))
                place_type = str(props.get("type", props.get("osm_value", "unknown"))).lower()
                full_text = " ".join([name, city_prop, state_prop, country_prop, str(props.get("street", ""))]).lower()
                score = 0
                if locality.lower() in full_text: score += 12
                if city.lower() in full_text: score += 8
                if state.lower() in full_text: score += 5
                if country_prop.lower() in {"india", ""}: score += 2
                if place_type in {"city", "town", "village", "suburb", "neighbourhood", "district", "locality"}: score += 3
                ranked.append((score, feature, coords))
            if ranked:
                ranked.sort(key=lambda x: x[0], reverse=True)
                score, best, coords = ranked[0]
                props = best.get("properties", {}) or {}
                # Avoid accepting a result that clearly belongs to a different state/country.
                result_state = str(props.get("state", "")).lower()
                result_country = str(props.get("country", "")).lower()
                if (not result_state or state.lower() in result_state or result_state in state.lower()) and (not result_country or "india" in result_country):
                    is_city_level = locality.lower() == city.lower() or (city.lower() in str(props.get("name", "")).lower() and locality.lower() not in str(props.get("name", "")).lower())
                    return {"lat": float(coords[1]), "lon": float(coords[0]),
                            "display_name": props.get("name", query) + ", " + ", ".join(x for x in [props.get("city", ""), props.get("state", ""), props.get("country", "India")] if x),
                            "place_type": str(props.get("type", props.get("osm_value", "unknown"))),
                            "place_class": str(props.get("osm_type", "photon")),
                            "source": ("City-level fallback (Photon)" if query_index == 3 and locality.lower() != city.lower() else "Photon geocoding"),
                            "coarse": (query_index == 3 and locality.lower() != city.lower()) or is_city_level}
        except Exception:
            pass

    # Nominatim backup. Use multiple query forms and do not reject every result just because
    # its type is not in a narrow list; matching the requested name/state is more important.
    nominatim_queries = list(dict.fromkeys([
        f"{locality}, {city}, {state}, India", f"{locality}, {city}, India",
        f"{locality}, {state}, India", f"{city}, {state}, India",
    ]))
    for query in nominatim_queries:
        try:
            response = requests.get("https://nominatim.openstreetmap.org/search", params={
                "q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1
            }, headers=HEADERS, timeout=(3, 8))
            response.raise_for_status()
            for item in response.json():
                display = str(item.get("display_name", ""))
                address = item.get("address", {}) or {}
                text_all = (display + " " + " ".join(str(v) for v in address.values())).lower()
                # Match state/country and requested locality or city to reduce false matches.
                if state.lower() not in text_all and "kerala" not in text_all and state.lower() != "":
                    continue
                if "india" not in text_all:
                    continue
                name_match = locality.lower() in text_all or city.lower() in text_all
                if not name_match:
                    continue
                return {"lat": float(item["lat"]), "lon": float(item["lon"]),
                        "display_name": display, "place_type": item.get("type", "unknown"),
                        "place_class": item.get("class", "unknown"), "source": "Nominatim geocoding",
                        "coarse": locality.lower() == city.lower()}
        except Exception:
            continue

    # Only explicitly configured locality fallback points are used here.
    fb = fallback_location(locality, city, state)
    if fb:
        return {**fb, "source": "Configured locality fallback", "coarse": False}
    return None



@st.cache_data(ttl=1800, show_spinner=False)
def location_suggestions(query, state, city="", mode="locality"):
    """Autocomplete locality/city names using Nominatim search; only place-type results."""
    query = (query or "").strip()
    if len(query) < 3:
        return []
    if mode == "city":
        search_query = f"{query}, {state}, India"
    else:
        search_query = f"{query}, {city}, {state}, India" if city else f"{query}, {state}, India"
    allowed_types = {"city", "town", "village", "suburb", "neighbourhood", "quarter", "city_district", "locality", "municipality", "administrative"}
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search", params={
            "q": search_query, "format": "jsonv2", "limit": 7, "addressdetails": 1, "dedupe": 1
        }, headers=HEADERS, timeout=(3, 7))
        response.raise_for_status()
        suggestions = []
        for item in response.json():
            typ = str(item.get("type", "")).lower()
            cls = str(item.get("class", "")).lower()
            if cls in {"shop", "amenity", "building", "office", "tourism", "leisure", "craft"}:
                continue
            if typ not in allowed_types and cls != "place":
                continue
            addr = item.get("address") or {}
            place_name = addr.get("neighbourhood") or addr.get("suburb") or addr.get("city_district") or addr.get("town") or addr.get("city") or addr.get("village") or item.get("name") or item.get("display_name", "").split(",")[0]
            city_name = addr.get("city") or addr.get("town") or addr.get("municipality") or addr.get("county") or city or ""
            state_name = addr.get("state") or state
            label = item.get("display_name", place_name)
            # Avoid duplicate labels while preserving candidate order.
            if not any(x["label"] == label for x in suggestions):
                suggestions.append({"label": label, "place": place_name, "city": city_name, "state": state_name,
                    "lat": item.get("lat"), "lon": item.get("lon"), "type": typ})
        return suggestions
    except Exception:
        return []

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0, 1-a)))


def tags_to_place(tags, lat=None, lon=None, distance_km=None):
    tags = tags or {}
    category = tags.get("shop") or tags.get("amenity") or tags.get("craft") or tags.get("healthcare") or tags.get("leisure") or tags.get("tourism") or tags.get("office") or "other"
    return {"name": tags.get("name", ""), "shop": tags.get("shop", ""), "amenity": tags.get("amenity", ""),
            "craft": tags.get("craft", ""), "healthcare": tags.get("healthcare", ""), "category": category,
            "latitude": lat, "longitude": lon, "distance_km": distance_km, "tags": tags}


# Layer-level status distinguishes a completed zero from a failed request.
OSM_LAYER_STATUS = {"shops": None, "food": None, "education": None, "health": None}
OSM_LAYER_ERRORS = {}


@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_osm_combined(lat, lon, radius_m):
    """One combined Overpass request; cached for 6 hours to avoid repeated waits."""
    lat, lon, radius_m = float(lat), float(lon), int(radius_m)
    query = f"""[out:json][timeout:12];
(
  nwr(around:{radius_m},{lat},{lon})["shop"];
  nwr(around:{radius_m},{lat},{lon})["craft"];
  nwr(around:{radius_m},{lat},{lon})["amenity"~"^(cafe|restaurant|fast_food|food_court|bakery|marketplace|school|college|university|kindergarten|hospital|clinic|pharmacy|doctors|dentist)$"];
  nwr(around:{radius_m},{lat},{lon})["healthcare"~"^(hospital|clinic|pharmacy|doctor|dentist)$"];
  nwr(around:{radius_m},{lat},{lon})["building"="school"];
);
out center tags;"""
    errors = []
    # Two attempts only. The previous implementation could make 12 sequential
    # requests (4 layers x 3 mirrors), each with a long timeout.
    for endpoint in OVERPASS_ENDPOINTS[:2]:
        try:
            response = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=(3, 10)
            )
            response.raise_for_status()
            elements = response.json().get("elements")
            if not isinstance(elements, list):
                raise ValueError("Overpass response did not contain an elements list")
            places = []
            seen = set()
            for element in elements:
                tags = element.get("tags") or {}
                if element.get("type") == "node":
                    place_lat, place_lon = element.get("lat"), element.get("lon")
                else:
                    center = element.get("center") or {}
                    place_lat, place_lon = center.get("lat"), center.get("lon")
                if place_lat is None or place_lon is None:
                    continue
                place_lat, place_lon = float(place_lat), float(place_lon)
                distance = haversine_km(lat, lon, place_lat, place_lon)
                if distance > radius_m / 1000:
                    continue
                key = (element.get("type"), element.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                places.append(tags_to_place(tags, place_lat, place_lon, distance))
            return places, None, endpoint
        except requests.exceptions.Timeout:
            errors.append(f"{endpoint}: timeout")
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "unknown"
            errors.append(f"{endpoint}: HTTP {code}")
        except (requests.exceptions.RequestException, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    return None, " | ".join(errors), None


def live_osm_places(lat, lon, radius_m):
    """Fast combined OSM fetch with conservative layer-status reporting."""
    global OSM_LAYER_STATUS, OSM_LAYER_ERRORS
    places, error, server = _fetch_osm_combined(float(lat), float(lon), int(radius_m))
    if places is None:
        OSM_LAYER_STATUS = {"shops": False, "food": False, "education": False, "health": False}
        OSM_LAYER_ERRORS = {"combined": error or "OSM request failed"}
        return None, "OSM request failed. " + (error or ""), None

    # A successful combined query means all requested tag groups were queried.
    # A zero match is a completed query, not proof that no facility exists.
    OSM_LAYER_STATUS = {"shops": True, "food": True, "education": True, "health": True}
    OSM_LAYER_ERRORS = {}
    return places, None, server

def normalize_osm_row(row):
    """Normalize legacy and v2 cache formats, including JSON tag payloads."""
    def first(names):
        for n in names:
            if n in row.index and pd.notna(row[n]):
                val = str(row[n]).strip()
                if val and val.lower() != "nan": return val
        return ""
    # Older cache files sometimes store original OSM tags only in a JSON Tags column.
    raw_tags = {}
    raw_json = first(["Tags", "tags", "OSM_Tags"])
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict): raw_tags = parsed
        except (json.JSONDecodeError, TypeError):
            raw_tags = {}
    shop = first(["shop", "Shop_Type", "shop_type"]) or str(raw_tags.get("shop", ""))
    amenity = first(["amenity"]) or str(raw_tags.get("amenity", ""))
    craft = first(["craft"]) or str(raw_tags.get("craft", ""))
    healthcare = first(["healthcare"]) or str(raw_tags.get("healthcare", ""))
    category = first(["category", "Category", "Business_Category", "business_category"])
    name = first(["name", "Business_Name", "business_name", "name:en"]) or str(raw_tags.get("name", ""))
    # Infer facility tags from generic category fields when separate OSM tags were not saved.
    normalized_category = category.lower().replace(" ", "_")
    combined = " ".join([shop, amenity, healthcare, normalized_category]).lower().replace("_", " ")
    if not amenity:
        if any(term in combined for term in ["school", "college", "university", "kindergarten", "education"]):
            amenity = next((term for term in ["school", "college", "university", "kindergarten"] if term in combined), "school")
        elif any(term in combined for term in ["hospital", "clinic", "medical center", "medical_centre"]):
            amenity = "hospital" if "hospital" in combined else "clinic"
        elif any(term in combined for term in ["doctor", "doctors"]):
            amenity = "doctors"
        elif "dentist" in combined:
            amenity = "dentist"
    if not healthcare and any(term in combined for term in ["hospital", "clinic", "doctor", "medical"]):
        healthcare = "hospital" if "hospital" in combined else "clinic" if "clinic" in combined else "doctor"
    if not shop and category and not amenity and not healthcare:
        shop = category
    return {"name": name, "shop": shop.lower().replace(" ", "_"), "amenity": amenity.lower(),
            "craft": craft.lower(), "healthcare": healthcare.lower(), "category": normalized_category,
            "latitude": first(["Latitude", "latitude", "lat"]), "longitude": first(["Longitude", "longitude", "lon", "long"]),
            "distance_km": first(["Distance_KM", "distance_km"]), "tags": raw_tags}


def local_cache_places(lat, lon, radius_m, locality, city, state):
    """Use geographic matching first; prefer exact metadata when available."""
    global OSM_LAYER_STATUS, OSM_LAYER_ERRORS
    errors = []
    for path in OSM_CACHE_PATHS:
        if not path.exists():
            continue
        try:
            original = pd.read_csv(path, low_memory=False)
            lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in original.columns), None)
            lon_col = next((c for c in ["Longitude", "longitude", "lon", "long"] if c in original.columns), None)
            if not lat_col or not lon_col:
                errors.append(f"{path.name}: no coordinate columns"); continue
            df = original.copy()
            df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
            df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
            df = df.dropna(subset=[lat_col, lon_col]).copy()
            if df.empty:
                errors.append(f"{path.name}: no valid coordinates"); continue
            df["_dist"] = df.apply(lambda r: haversine_km(lat, lon, r[lat_col], r[lon_col]), axis=1)
            spatial = df[df["_dist"] <= radius_m / 1000].copy()
            if spatial.empty:
                errors.append(f"{path.name}: no records in selected radius"); continue
            if {"Locality", "City", "State"}.issubset(spatial.columns):
                exact = spatial[
                    spatial["Locality"].astype(str).str.strip().str.casefold().eq(str(locality).strip().casefold()) &
                    spatial["City"].astype(str).str.strip().str.casefold().eq(str(city).strip().casefold()) &
                    spatial["State"].astype(str).str.strip().str.casefold().eq(str(state).strip().casefold())]
                if not exact.empty:
                    spatial = exact.copy()
            places = []
            for _, row in spatial.iterrows():
                item = normalize_osm_row(row)
                item["latitude"], item["longitude"], item["distance_km"] = float(row[lat_col]), float(row[lon_col]), float(row["_dist"])
                places.append(item)
            layer_cols = {"Layer_Shops": "shops", "Layer_Food": "food", "Layer_Education": "education", "Layer_Health": "health"}
            if all(col in spatial.columns for col in layer_cols):
                row = spatial.iloc[-1]
                OSM_LAYER_STATUS = {layer: str(row[col]).strip().lower() in {"true", "1", "yes"} for col, layer in layer_cols.items()}
            else:
                # Legacy cache has no completeness flags. A positive tagged match is
                # usable as a mapped match; absence remains unknown, never a verified zero.
                OSM_LAYER_STATUS = {
                    "shops": True if any(p.get("shop") or p.get("craft") for p in places) else None,
                    "food": True if any(str(p.get("amenity", "")).lower() in {"cafe", "restaurant", "fast_food", "food_court", "bakery", "marketplace"} for p in places) else None,
                    "education": True if any(is_school_place(p) for p in places) else None,
                    "health": True if any(is_health_place(p) for p in places) else None,
                }
            OSM_LAYER_ERRORS = {}
            return places, f"Using {len(places)} geographically matched cached records from {path.name}."
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    OSM_LAYER_STATUS = {"shops": None, "food": None, "education": None, "health": None}
    OSM_LAYER_ERRORS = {}
    return None, "No matching cached OSM records found. " + " | ".join(errors)

def _clean_col_name(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def uploaded_csv_places(uploaded_files, lat, lon, radius_m, locality, city, state):
    """Read user-supplied geocoded CSVs and return only rows within the chosen radius.

    CSV rows are treated as supplied records, not as an exhaustive census. Missing
    layers remain unknown; a zero is never inferred from an empty or absent file.
    """
    global OSM_LAYER_STATUS
    all_places = []
    layer_has_rows = {"shops": False, "food": False, "education": False, "health": False}
    notes = []
    survey_frames = []

    for label, uploaded in uploaded_files:
        if uploaded is None:
            continue
        try:
            uploaded.seek(0)
            frame = pd.read_csv(uploaded, low_memory=False)
            frame.columns = [_clean_col_name(c) for c in frame.columns]
            if label == "Customer survey":
                # Survey rows are read separately for this analysis and are not counted as POIs.
                survey_frames.append(frame)
                notes.append(f"{label}: {len(frame):,} rows loaded (survey evidence; not counted as mapped facilities)")
                continue

            lat_col = next((c for c in ("latitude", "lat") if c in frame.columns), None)
            lon_col = next((c for c in ("longitude", "lon", "long") if c in frame.columns), None)
            if not lat_col or not lon_col:
                notes.append(f"{label}: skipped — requires latitude and longitude columns")
                continue
            frame[lat_col] = pd.to_numeric(frame[lat_col], errors="coerce")
            frame[lon_col] = pd.to_numeric(frame[lon_col], errors="coerce")
            frame = frame.dropna(subset=[lat_col, lon_col]).copy()
            if frame.empty:
                notes.append(f"{label}: skipped — no valid coordinates")
                continue

            # If locality metadata exists, use it as an additional guard. Coordinates
            # and radius remain the primary spatial filter.
            if "state" in frame.columns:
                frame = frame[frame["state"].astype(str).str.casefold().str.contains(str(state).casefold(), na=False)]
            if "city" in frame.columns:
                frame = frame[frame["city"].astype(str).str.casefold().str.contains(str(city).casefold(), na=False)]
            if "locality" in frame.columns:
                locality_match = frame["locality"].astype(str).str.casefold().str.contains(str(locality).casefold(), na=False)
                # Keep nearby coordinate matches if locality is not provided exactly;
                # do not require a locality column for valid geocoded files.
                if locality_match.any():
                    frame = frame[locality_match]

            added = 0
            for _, row in frame.iterrows():
                row_lat, row_lon = float(row[lat_col]), float(row[lon_col])
                distance = haversine_km(float(lat), float(lon), row_lat, row_lon)
                if distance > float(radius_m) / 1000:
                    continue
                raw = {str(k): ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                name = next((raw.get(k, "") for k in ("name", "business_name", "school_name", "hospital_name", "facility_name") if raw.get(k)), "")
                category = next((raw.get(k, "") for k in ("category", "business_category", "shop_category", "facility_type", "type") if raw.get(k)), "")
                place = tags_to_place({}, row_lat, row_lon, distance)
                place["name"] = name
                place["category"] = category.strip().lower().replace(" ", "_")
                if label in {"Government / MSME businesses", "Business directory"}:
                    place["shop"] = (raw.get("shop") or raw.get("business_category") or raw.get("category") or raw.get("type") or "").lower().replace(" ", "_")
                    place["craft"] = (raw.get("craft") or "").lower()
                    layer_has_rows["shops"] = True
                    layer_has_rows["food"] = True
                    layer_has_rows["health"] = True
                elif label == "Schools / colleges":
                    place["amenity"] = "school"
                    place["category"] = "school"
                    layer_has_rows["education"] = True
                elif label == "Hospitals / clinics":
                    place["amenity"] = "hospital"
                    place["healthcare"] = "hospital"
                    place["category"] = "hospital"
                    layer_has_rows["health"] = True
                all_places.append(place)
                added += 1
            notes.append(f"{label}: {added:,} rows within selected radius")
        except Exception as exc:
            notes.append(f"{label}: could not read CSV ({type(exc).__name__}: {exc})")

    survey_count = sum(len(x) for x in survey_frames)
    return all_places, layer_has_rows, notes, survey_count


def merge_place_records(*groups):
    """Merge POI lists while removing obvious duplicate name/coordinate rows."""
    merged = {}
    for group in groups:
        for item in (group or []):
            lat = item.get("latitude")
            lon = item.get("longitude")
            name = str(item.get("name", "")).strip().casefold()
            category = str(item.get("category", item.get("shop", item.get("amenity", "")))).casefold()
            if lat is None or lon is None:
                key = (name, category)
            else:
                key = (name, category, round(float(lat), 5), round(float(lon), 5))
            merged[key] = item
    return list(merged.values())


def save_live_cache(places, locality, city, state, radius_m, layer_status=None):
    """Append/update records in a separate v2 CSV; preserve legacy cache files."""
    if not places: return 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = []
    for p in places:
        tags = p.get("tags") or {}
        rows.append({"Locality": locality, "City": city, "State": state, "Country": "India",
                     "Business_Name": p.get("name", ""), "Category": p.get("category", ""),
                     "shop": p.get("shop", ""), "amenity": p.get("amenity", ""),
                     "craft": p.get("craft", ""), "healthcare": p.get("healthcare", ""),
                     "Latitude": p.get("latitude"), "Longitude": p.get("longitude"),
                     "Distance_KM": p.get("distance_km"), "Radius_Meters": radius_m,
                     "Data_Status": "LIVE OSM", "Timestamp": now, "Tags": json.dumps(tags, ensure_ascii=False),
                     "Layer_Shops": bool((layer_status or {}).get("shops", False)),
                     "Layer_Food": bool((layer_status or {}).get("food", False)),
                     "Layer_Education": bool((layer_status or {}).get("education", False)),
                     "Layer_Health": bool((layer_status or {}).get("health", False))})
    new_df = pd.DataFrame(rows)
    if CACHE_V2.exists():
        try:
            old = pd.read_csv(CACHE_V2, low_memory=False)
            if not old.empty:
                # Remove duplicate entries for this locality based on name + coordinates.
                if {"Locality", "City", "State"}.issubset(old.columns):
                    old = old[~((old["Locality"].astype(str).str.lower() == locality.lower()) &
                                (old["City"].astype(str).str.lower() == city.lower()) &
                                (old["State"].astype(str).str.lower() == state.lower()))]
                new_df = pd.concat([old, new_df], ignore_index=True)
        except Exception:
            pass
    new_df = new_df.drop_duplicates(subset=["Locality", "City", "State", "Business_Name", "Latitude", "Longitude"], keep="last")
    new_df.to_csv(CACHE_V2, index=False, encoding="utf-8-sig")
    return len(new_df)


def category_matches(category, tags):
    shop = str(tags.get("shop", "")).strip().lower().replace(" ", "_")
    amenity = str(tags.get("amenity", "")).strip().lower()
    craft = str(tags.get("craft", "")).strip().lower()
    healthcare = str(tags.get("healthcare", "")).strip().lower()
    generic = str(tags.get("category", "")).strip().lower().replace(" ", "_")
    shop = shop or generic
    matches = {
        "Grocery": shop in {"supermarket", "convenience", "grocery", "general", "greengrocer", "food", "deli", "variety_store", "department_store", "farm", "agrarian"},
        "Cafe": amenity == "cafe" or shop == "coffee" or generic == "cafe",
        "Stationery": shop in {"stationery", "books", "copyshop", "printing", "office_supplies"} or "stationery" in generic,
        "Laundry": shop in {"laundry", "dry_cleaning"} or craft in {"laundry", "dry_cleaning"},
        "Computer Repair": shop in {"computer", "electronics", "mobile_phone", "computer_repair", "electronics_repair"} or craft in {"electronics_repair", "computer_repair"},
        "Pharmacy": amenity == "pharmacy" or healthcare == "pharmacy" or shop in {"chemist", "pharmacy", "medical_supply"} or generic in {"pharmacy", "chemist"},
        "Bakery": shop == "bakery" or amenity == "bakery" or generic == "bakery",
        "Clothing": shop in {"clothes", "fashion", "boutique", "fabric", "tailor", "sewing", "shoes", "leather", "watches", "jewelry", "jewellery"} or craft == "tailor",
        "Restaurant": amenity in {"restaurant", "fast_food", "food_court"} or generic in {"restaurant", "fast_food", "food_court"},
    }
    return bool(matches.get(category, False))


def rf_predict_demand(model, category, locality, population, competitors, schools, hospitals, rent):
    if model is None or not hasattr(model, "feature_names_in_"): return None
    row = {}
    for col in list(model.feature_names_in_):
        key = str(col).strip().lower()
        if key in {"shop_category", "category"}: value = category
        elif key in {"neighbourhood", "neighborhood", "locality", "location"}: value = locality
        elif "population" in key: value = population
        elif "competitor" in key: value = competitors
        elif "monthly_rent" in key or key == "rent": value = rent
        elif "school" in key: value = schools
        elif "hospital" in key or "clinic" in key: value = hospitals
        else: return None
        row[col] = value
    try:
        prediction = float(model.predict(pd.DataFrame([row], columns=list(model.feature_names_in_)))[0])
        if not np.isfinite(prediction): return None
        return max(0, int(round(prediction)))
    except Exception:
        return None


def estimate_demand(category, population, competitor_count, school_count, hospital_count):
    profile = PROFILES[category]
    pop_factor = max(.25, min(3.0, population / 10000)) if population > 0 else 1.0
    facility_factor = 1.0 if school_count is None or hospital_count is None else 1 + min(.25, school_count*.01 + hospital_count*.02)
    competition_factor = 1.0 if competitor_count is None else 1 / (1 + max(0, competitor_count)*.12)
    return max(0, int(round(profile["base"] * pop_factor * facility_factor * competition_factor)))


def is_school_place(p):
    amenity = str(p.get("amenity", "")).lower()
    category = str(p.get("category", "")).lower().replace("_", " ")
    shop = str(p.get("shop", "")).lower().replace("_", " ")
    return amenity in {"school", "college", "university", "kindergarten"} or any(
        term in category or term in shop for term in ["school", "college", "university", "kindergarten", "education"])


def is_health_place(p):
    amenity = str(p.get("amenity", "")).lower()
    healthcare = str(p.get("healthcare", "")).lower()
    category = str(p.get("category", "")).lower().replace("_", " ")
    shop = str(p.get("shop", "")).lower().replace("_", " ")
    medical_terms = ["hospital", "clinic", "doctor", "doctors", "dentist", "medical"]
    return amenity in {"hospital", "clinic", "doctors", "doctor", "dentist"} or healthcare in {
        "hospital", "clinic", "doctor", "doctors", "dentist"} or any(
        term in category or term in shop for term in medical_terms)


def make_comparison(categories, model, locality, population, places, rent, budget):
    status = OSM_LAYER_STATUS.copy()
    valid = places is not None
    schools = sum(is_school_place(p) for p in places) if valid and status.get("education") is True else None
    hospitals = sum(is_health_place(p) for p in places) if valid and status.get("health") is True else None
    layer_for_category = {"Grocery":"shops", "Stationery":"shops", "Laundry":"shops", "Computer Repair":"shops", "Clothing":"shops", "Cafe":"food", "Restaurant":"food", "Bakery":"food", "Pharmacy":"health"}
    rows = []
    for cat in categories:
        layer = layer_for_category.get(cat, "shops")
        competitors = sum(category_matches(cat, p) for p in places) if valid and status.get(layer) is True else None
        demand = None
        if valid and competitors is not None and schools is not None and hospitals is not None:
            demand = rf_predict_demand(model, cat, locality, population, competitors, schools, hospitals, rent)
        method = "Saved Random Forest (preliminary)" if demand is not None else "Rule-based fallback; not an ML prediction"
        if demand is None:
            demand = estimate_demand(cat, population, competitors, schools, hospitals)
        profile = PROFILES[cat]
        revenue = demand * profile["avg_bill"]
        cogs = revenue * profile["cogs"]
        fixed = profile["staff"] + profile["utilities"] + profile["marketing"] + rent
        total_cost = cogs + fixed
        profit = revenue - total_cost
        upfront = profile["setup"] + profile["stock"] + 4 * rent
        contribution = profile["avg_bill"] * (1-profile["cogs"])
        breakeven = math.ceil(fixed/contribution) if contribution > 0 else None
        margin = profit/revenue*100 if revenue > 0 else 0
        payback = upfront/profit if profit > 0 else None
        rows.append({"Shop category":cat, "Mapped competitors in radius":competitors, "Mapped schools":schools, "Mapped hospitals/clinics":hospitals, "Estimated monthly transactions":demand, "Estimated monthly revenue (₹)":round(revenue), "Estimated monthly operating cost (₹)":round(total_cost), "Estimated monthly operating profit (₹)":round(profit), "Estimated profit margin (%)":round(margin,1), "Break-even transactions/month (approx.)":breakeven, "Estimated upfront investment (₹)":round(upfront), "Estimated payback period (months)":round(payback,1) if payback is not None else "Not reached at estimated profit", "Within budget":"Yes" if upfront <= budget else "No", "Demand method":method})
    df = pd.DataFrame(rows)
    df["_budget_sort"] = df["Within budget"].map({"Yes":0,"No":1})
    return df.sort_values(["_budget_sort","Estimated monthly operating profit (₹)"], ascending=[True,False]).drop(columns="_budget_sort").reset_index(drop=True)

def add_suitability_scores(result, places_available, priority):
    df = result.copy()
    def scaled(series, higher=True):
        s = pd.to_numeric(series, errors="coerce")
        if s.isna().all(): return pd.Series(50.0, index=df.index)
        lo, hi = s.min(), s.max()
        score = pd.Series(50.0, index=df.index) if hi == lo else (s-lo)/(hi-lo)*100
        return score if higher else 100-score
    demand_score = scaled(df["Estimated monthly transactions"])
    profit_score = scaled(df["Estimated monthly operating profit (₹)"])
    budget_score = df["Within budget"].map({"Yes":100.0,"No":0.0}).fillna(0)
    competition_score = scaled(df["Mapped competitors in radius"], False) if places_available and df["Mapped competitors in radius"].notna().any() else pd.Series(50.0,index=df.index)
    presets = {"Balanced (recommended)":(.30,.30,.20,.10,.10), "Demand priority":(.50,.20,.10,.10,.10),
              "Profit priority":(.20,.50,.10,.10,.10), "Low competition priority":(.20,.20,.10,.40,.10),
              "Budget-fit priority":(.20,.20,.10,.10,.40)}
    wd,wp,wloc,wc,wb = presets.get(priority,presets["Balanced (recommended)"])
    location_score = pd.Series(50.0,index=df.index)
    df["Demand Score"] = demand_score.round(1); df["Profit Score"] = profit_score.round(1)
    df["Competition Score"] = competition_score.round(1); df["Budget Score"] = budget_score.round(1)
    df["Suitability Score"] = (wd*demand_score+wp*profit_score+wloc*location_score+wc*competition_score+wb*budget_score).round(1)
    df["_eligible"] = df["Within budget"].map({"Yes":1,"No":0}).fillna(0)
    return df.sort_values(["_eligible","Suitability Score","Estimated monthly operating profit (₹)"],ascending=[False,False,False]).drop(columns="_eligible").reset_index(drop=True)


# ----------------------------- APP UI -----------------------------
st.title("🇮🇳 India Shop Business Advisor")
st.write("Compare shop ideas for a selected Indian locality using investment budget, expected rent and nearby mapped places.")
st.info("OpenStreetMap coverage is incomplete and does not measure demand. The saved Random Forest may be trained on a small illustrative dataset. Average bill, setup cost, expenses and rule-based demand are planning assumptions.")

# Predefined city and neighbourhood dropdowns (curated starter lists; extend as needed).
CITY_LISTS = {
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur", "Malappuram", "Kannur", "Kollam", "Alappuzha", "Palakkad", "Kottayam", "Kasaragod", "Wayanad", "Pathanamthitta", "Idukki"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi", "Kalaburagi", "Davanagere", "Udupi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Vellore", "Thanjavur", "Tiruppur"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane", "Aurangabad", "Kolhapur", "Navi Mumbai"],
    "Delhi": ["New Delhi", "Delhi"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Nellore", "Kurnool"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Noida", "Ghaziabad", "Prayagraj"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Karnal", "Ambala"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bhilai"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Rishikesh", "Haldwani"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Manali", "Solan"],
    "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag"],
    "Tripura": ["Agartala"], "Manipur": ["Imphal"], "Meghalaya": ["Shillong"],
    "Arunachal Pradesh": ["Itanagar"], "Nagaland": ["Kohima", "Dimapur"],
    "Mizoram": ["Aizawl"], "Sikkim": ["Gangtok"],
    "Andaman and Nicobar Islands": ["Port Blair"], "Chandigarh": ["Chandigarh"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Daman", "Silvassa"],
    "Ladakh": ["Leh"], "Lakshadweep": ["Kavaratti"], "Puducherry": ["Puducherry"],
}
# Add starter cities for states that were not in the initial curated city mapping.
CITY_LISTS.update({
    "Arunachal Pradesh": ["Itanagar", "Naharlagun", "Pasighat"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bhilai", "Korba"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Manali", "Solan", "Mandi"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Hazaribagh"],
    "Manipur": ["Imphal", "Thoubal", "Churachandpur"],
    "Meghalaya": ["Shillong", "Tura", "Jowai"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung"],
    "Sikkim": ["Gangtok", "Namchi", "Gyalshing"],
    "Tripura": ["Agartala", "Udaipur", "Dharmanagar"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Rishikesh", "Haldwani", "Roorkee"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri", "Asansol"],
    "Andaman and Nicobar Islands": ["Port Blair", "Diglipur"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Daman", "Silvassa", "Dadra"],
    "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag", "Baramulla"],
    "Ladakh": ["Leh", "Kargil"],
    "Lakshadweep": ["Kavaratti", "Agatti"],
    "Puducherry": ["Puducherry", "Karaikal", "Mahe", "Yanam"],
})

NEIGHBOURHOOD_LISTS = {
    "Kochi": ["Kakkanad", "Edappally", "Aluva", "Vyttila", "Palarivattom", "Fort Kochi", "Kaloor", "Thrippunithura", "Marine Drive", "Mattancherry", "Thevara", "Kadavanthra", "Ernakulam North", "Ernakulam South", "Infopark", "Thrikkakara"],
    "Thiruvananthapuram": ["Technopark", "Kazhakkoottam", "Pattom", "Kowdiar", "Thampanoor", "Vellayambalam", "Neyyattinkara", "Attingal", "Sreekaryam", "Karamana"],
    "Kozhikode": ["Kozhikode Beach", "Mavoor Road", "Nadakkavu", "Kallai", "Feroke", "Kunnamangalam", "Medical College", "Ramanattukara"],
    "Thrissur": ["Ayyanthole", "Punkunnam", "West Fort", "Ollur", "Guruvayur", "Kodungallur", "Irinjalakuda"],
    "Malappuram": ["Tirur", "Manjeri", "Perinthalmanna", "Kottakkal", "Ponnani", "Nilambur", "Tirurangadi", "Thalappara"],
    "Kannur": ["Thalassery", "Payyannur", "Taliparamba", "Kannur Town", "Mattannur"],
    "Kollam": ["Kollam Town", "Karunagappally", "Kottarakkara", "Punalur"],
    "Alappuzha": ["Alappuzha Town", "Cherthala", "Kayamkulam", "Haripad"],
    "Palakkad": ["Palakkad Town", "Ottapalam", "Shoranur", "Chittur"],
    "Kottayam": ["Kottayam Town", "Changanassery", "Pala", "Ettumanoor"],
    "Bengaluru": ["Whitefield", "Electronic City", "Indiranagar", "Koramangala", "Jayanagar", "Hebbal", "Marathahalli", "HSR Layout", "Yelahanka", "Rajajinagar"],
    "Mysuru": ["Vijayanagar", "Hebbal", "Nazarbad", "Kuvempunagar", "Jayalakshmipuram"],
    "Mangaluru": ["Kadri", "Bejai", "Bendoor", "Surathkal", "Kankanady"],
    "Chennai": ["T. Nagar", "Anna Nagar", "Adyar", "Velachery", "Porur", "Tambaram", "OMR", "Mylapore", "Guindy", "Chromepet"],
    "Coimbatore": ["RS Puram", "Gandhipuram", "Peelamedu", "Saravanampatti", "Singanallur", "Town Hall"],
    "Madurai": ["Anna Nagar", "KK Nagar", "Tallakulam", "Simmakkal", "Thirunagar"],
    "Tiruchirappalli": ["Srirangam", "Thillai Nagar", "Cantonment", "Woraiyur", "Rockfort"],
    "Mumbai": ["Andheri", "Bandra", "Borivali", "Dadar", "Powai", "Thane", "Colaba", "Goregaon", "Kurla", "Chembur"],
    "Pune": ["Hinjawadi", "Kothrud", "Baner", "Viman Nagar", "Hadapsar", "Wakad", "Pimpri", "Aundh"],
    "Nagpur": ["Dharampeth", "Sitabuldi", "Manish Nagar", "Sadar"],
    "Hyderabad": ["Madhapur", "Hitech City", "Gachibowli", "Kukatpally", "Banjara Hills", "Jubilee Hills", "Secunderabad", "Ameerpet"],
    "Visakhapatnam": ["Madhurawada", "Gajuwaka", "Dwaraka Nagar", "Rushikonda", "Seethammadhara"],
    "Kolkata": ["Salt Lake", "New Town", "Park Street", "Behala", "Dum Dum", "Howrah", "Garia"],
    "Ahmedabad": ["Satellite", "Bopal", "Maninagar", "Navrangpura", "Prahlad Nagar", "Gota"],
    "Surat": ["Adajan", "Vesu", "Varachha", "Katargam", "Udhna"],
    "Jaipur": ["Malviya Nagar", "Vaishali Nagar", "Mansarovar", "C-Scheme", "Jagatpura"],
    "Lucknow": ["Gomti Nagar", "Hazratganj", "Aliganj", "Indira Nagar", "Aminabad"],
    "New Delhi": ["Connaught Place", "Dwarka", "Rohini", "Saket", "Lajpat Nagar", "Karol Bagh", "Vasant Kunj"],
    "Gurugram": ["Cyber City", "Sector 29", "Golf Course Road", "Sohna Road", "Sector 56", "Udyog Vihar"],
    "Noida": ["Sector 18", "Sector 62", "Sector 50", "Sector 137", "Sector 15", "Greater Noida West"],
    "Indore": ["Vijay Nagar", "Palasia", "Rajwada", "Bhawarkua", "Scheme 54"],
    "Bhopal": ["MP Nagar", "Arera Colony", "New Market", "Kolar Road", "Habibganj"],
    "Bhubaneswar": ["Patia", "Jaydev Vihar", "Saheed Nagar", "Old Town", "Unit 4"],
    "Guwahati": ["Dispur", "Paltan Bazaar", "Beltola", "Six Mile", "Zoo Road"],
    "Panaji": ["Miramar", "Fontainhas", "Taleigao", "Panjim Market"],
    "Ranchi": ["Lalpur", "Harmu", "Morabadi", "Doranda"],
    "Raipur": ["Shankar Nagar", "Pandri", "Telibandha", "Civil Lines"],
    "Dehradun": ["Rajpur Road", "ISBT", "Clock Tower", "Vasant Vihar"],
    "Shimla": ["Mall Road", "Sanjauli", "New Shimla", "Summer Hill"],
    "Srinagar": ["Lal Chowk", "Rajbagh", "Hazratbal", "Bemina"],
    "Jammu": ["Gandhi Nagar", "Trikuta Nagar", "Jewel Chowk", "Bahu Plaza"],
    "Agartala": ["Agartala City", "Pratapgarh", "Bardowali"],
    "Imphal": ["Imphal East", "Imphal West", "Thangmeiband"],
    "Shillong": ["Police Bazaar", "Laitumkhrah", "Nongthymmai", "Mawlai"],
    "Itanagar": ["Naharlagun", "Itanagar Town"], "Kohima": ["Kohima Town", "High School Area"],
    "Dimapur": ["Dimapur Town", "Chumukedima"], "Aizawl": ["Aizawl City", "Bawngkawn", "Dawrpui"],
    "Gangtok": ["MG Marg", "Tadong", "Deorali"], "Port Blair": ["Aberdeen", "Phoenix Bay", "Garacharma"],
    "Chandigarh": ["Sector 17", "Sector 22", "Sector 35", "Manimajra"], "Daman": ["Nani Daman", "Moti Daman"],
    "Silvassa": ["Silvassa Town", "Vapi Road"], "Leh": ["Leh Town", "Choglamsar"],
    "Kavaratti": ["Kavaratti Town"], "Puducherry": ["White Town", "Lawspet", "Ariyankuppam", "Muthialpet"],
}


# Expanded curated dropdown coverage. These are starter lists, not exhaustive official boundaries.
CITY_LISTS.update({
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur", "Malappuram", "Kannur", "Kollam", "Alappuzha", "Palakkad", "Kottayam", "Kasaragod", "Wayanad", "Pathanamthitta", "Idukki", "Thalassery", "Manjeri", "Tirur", "Perinthalmanna", "Kottakkal", "Ponnani", "Nilambur", "Tirurangadi", "Thalassery", "Payyannur", "Taliparamba", "Kochi", "Aluva", "Changanassery", "Pala", "Muvattupuzha", "Thodupuzha", "Irinjalakuda", "Guruvayur", "Kodungallur", "Ottapalam", "Shoranur", "Koyilandy", "Vadakara", "Kalpetta", "Sulthan Bathery", "Attingal", "Neyyattinkara", "Varkala"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi", "Kalaburagi", "Davanagere", "Udupi", "Ballari", "Vijayapura", "Shivamogga", "Tumakuru", "Hassan", "Bidar", "Raichur", "Chikkamagaluru", "Kolar", "Mandya", "Chitradurga", "Karwar"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode", "Vellore", "Thanjavur", "Tiruppur", "Thoothukudi", "Dindigul", "Kanchipuram", "Nagercoil", "Hosur", "Cuddalore", "Karur", "Sivakasi", "Pudukkottai", "Pollachi", "Ramanathapuram", "Ooty", "Kumbakonam"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane", "Aurangabad", "Kolhapur", "Navi Mumbai", "Solapur", "Amravati", "Akola", "Latur", "Ahmednagar", "Jalgaon", "Sangli", "Satara", "Chhatrapati Sambhajinagar", "Ratnagiri"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Bhavnagar", "Jamnagar", "Junagadh", "Anand", "Nadiad", "Morbi", "Mehsana", "Navsari", "Valsad", "Bharuch", "Porbandar"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Noida", "Ghaziabad", "Prayagraj", "Meerut", "Bareilly", "Aligarh", "Moradabad", "Gorakhpur", "Jhansi", "Mathura", "Firozabad", "Saharanpur", "Ayodhya", "Gonda"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner", "Alwar", "Bharatpur", "Sikar", "Pali", "Sri Ganganagar", "Bhilwara", "Chittorgarh"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Nellore", "Kurnool", "Rajahmundry", "Kakinada", "Anantapur", "Kadapa", "Eluru", "Ongole", "Srikakulam", "Vizianagaram"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam", "Mahbubnagar", "Nalgonda", "Adilabad", "Siddipet", "Ramagundam"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri", "Asansol", "Bardhaman", "Kharagpur", "Haldia", "Malda", "Jalpaiguri"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Hoshiarpur", "Pathankot", "Moga", "Firozpur"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Karnal", "Ambala", "Hisar", "Rohtak", "Sonipat", "Panchkula", "Rewari"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga", "Purnia", "Bihar Sharif", "Ara", "Begusarai", "Katihar"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain", "Sagar", "Rewa", "Satna", "Dewas", "Ratlam"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri", "Sambalpur", "Berhampur", "Balasore", "Angul", "Jharsuguda"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Tezpur", "Tinsukia", "Nagaon", "Bongaigaon"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Hazaribagh", "Deoghar", "Giridih", "Ramgarh"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bhilai", "Korba", "Raigarh", "Jagdalpur", "Ambikapur"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Rishikesh", "Haldwani", "Roorkee", "Rudrapur", "Kashipur", "Almora"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Manali", "Solan", "Mandi", "Kullu", "Hamirpur", "Una", "Kangra"],
    "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag", "Baramulla", "Sopore", "Kathua", "Udhampur"],
    "Arunachal Pradesh": ["Itanagar", "Naharlagun", "Pasighat", "Tawang", "Ziro", "Bomdila"],
    "Manipur": ["Imphal", "Thoubal", "Churachandpur", "Bishnupur", "Kakching"],
    "Meghalaya": ["Shillong", "Tura", "Jowai", "Nongstoin", "Williamnagar"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Kolasib", "Serchhip"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Tuensang", "Wokha"],
    "Sikkim": ["Gangtok", "Namchi", "Gyalshing", "Mangan", "Singtam"],
    "Tripura": ["Agartala", "Udaipur", "Dharmanagar", "Kailashahar", "Belonia"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda", "Calangute"],
    "Andaman and Nicobar Islands": ["Port Blair", "Diglipur", "Mayabunder", "Rangat"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Daman", "Silvassa", "Dadra", "Diu"],
    "Puducherry": ["Puducherry", "Karaikal", "Mahe", "Yanam"],
    "Delhi": ["New Delhi", "Delhi", "Dwarka", "Rohini", "Shahdara", "Narela"],
    "Chandigarh": ["Chandigarh"], "Ladakh": ["Leh", "Kargil"], "Lakshadweep": ["Kavaratti", "Agatti", "Andrott", "Minicoy"],
})

# Extend neighbourhood/locality dropdown options for additional cities.
NEIGHBOURHOOD_LISTS.update({
    "Kochi": ["Kakkanad", "Edappally", "Aluva", "Vyttila", "Palarivattom", "Fort Kochi", "Kaloor", "Thrippunithura", "Marine Drive", "Mattancherry", "Thevara", "Kadavanthra", "Ernakulam North", "Ernakulam South", "Infopark", "Thrikkakara", "Maradu", "Kumbalam", "Pachalam", "Cheranallur", "Angamaly", "Perumbavoor", "North Paravur", "Tripunithura", "Willingdon Island"],
    "Thiruvananthapuram": ["Technopark", "Kazhakkoottam", "Pattom", "Kowdiar", "Thampanoor", "Vellayambalam", "Neyyattinkara", "Attingal", "Sreekaryam", "Karamana", "Kesavadasapuram", "Peroorkada", "Palayam", "Ulloor", "Vazhuthacaud", "Poojappura", "Balaramapuram", "Vizhinjam", "Aakulam"],
    "Kozhikode": ["Kozhikode Beach", "Mavoor Road", "Nadakkavu", "Kallai", "Feroke", "Kunnamangalam", "Medical College", "Ramanattukara", "Kozhikode City", "West Hill", "Vellimadukunnu", "Palazhi", "Kottooli", "Thondayad", "Beypore", "Vadakara", "Koyilandy"],
    "Thrissur": ["Ayyanthole", "Punkunnam", "West Fort", "Ollur", "Guruvayur", "Kodungallur", "Irinjalakuda", "East Fort", "Mannuthy", "Kuttanellur", "Cherpu", "Wadakkanchery", "Chalakudy"],
    "Malappuram": ["Tirur", "Manjeri", "Perinthalmanna", "Kottakkal", "Ponnani", "Nilambur", "Tirurangadi", "Thalappara", "Malappuram Town", "Edappal", " Kondotty", "Areekode", "Valanchery", "Parappanangadi"],
    "Kannur": ["Thalassery", "Payyannur", "Taliparamba", "Kannur Town", "Mattannur", "Chakkarakkal", "Pallikkunnu", "Thottada", "Iritty", "Kuthuparamba"],
    "Kollam": ["Kollam Town", "Karunagappally", "Kottarakkara", "Punalur", "Chavara", "Paravur", "Kundara", "Anchal", "Oachira"],
    "Alappuzha": ["Alappuzha Town", "Cherthala", "Kayamkulam", "Haripad", "Ambalappuzha", "Mavelikkara", "Aroor", "Thanneermukkom"],
    "Palakkad": ["Palakkad Town", "Ottapalam", "Shoranur", "Chittur", "Mannarkkad", "Pattambi", "Kuzhalmannam", "Kanjikode"],
    "Kottayam": ["Kottayam Town", "Changanassery", "Pala", "Ettumanoor", "Kanjirappally", "Vaikom", "Kuravilangad", "Kumarakom"],
    "Bengaluru": ["Whitefield", "Electronic City", "Indiranagar", "Koramangala", "Jayanagar", "Hebbal", "Marathahalli", "HSR Layout", "Yelahanka", "Rajajinagar", "BTM Layout", "JP Nagar", "Malleshwaram", "Banashankari", "Basavanagudi", "Sarjapur Road", "Bellandur", "Mahadevapura", "Brookefield", "Peenya", "Vijayanagar", "RT Nagar"],
    "Mysuru": ["Vijayanagar", "Hebbal", "Nazarbad", "Kuvempunagar", "Jayalakshmipuram", "Saraswathipuram", "Gokulam", "Hunsur Road", "Bannimantap", "Lashkar Mohalla"],
    "Mangaluru": ["Kadri", "Bejai", "Bendoor", "Surathkal", "Kankanady", "Hampankatta", "Mangaladevi", "Bondel", "Derebail", "Panambur"],
    "Hubballi": ["Vidyanagar", "Gokul Road", "Unkal", "Old Hubli", "Keshwapur", "Deshpande Nagar", "Navanagar"],
    "Chennai": ["T. Nagar", "Anna Nagar", "Adyar", "Velachery", "Porur", "Tambaram", "OMR", "Mylapore", "Guindy", "Chromepet", "Kodambakkam", "Nungambakkam", "Pallavaram", "Perambur", "Besant Nagar", "Sholinganallur", "Thiruvanmiyur", "Ambattur", "Avadi"],
    "Coimbatore": ["RS Puram", "Gandhipuram", "Peelamedu", "Saravanampatti", "Singanallur", "Town Hall", "Saibaba Colony", "Race Course", "Ukkadam", "Vadavalli", "Kuniyamuthur", "Avinashi Road", "Kalapatti"],
    "Madurai": ["Anna Nagar", "KK Nagar", "Tallakulam", "Simmakkal", "Thirunagar", "Goripalayam", "Mattuthavani", "Arapalayam", "Palanganatham", "Villapuram"],
    "Tiruchirappalli": ["Srirangam", "Thillai Nagar", "Cantonment", "Woraiyur", "Rockfort", "K.K. Nagar", "Anna Nagar", "Bharathidasan Nagar", "Puthur", "Tennur"],
    "Tiruppur": ["Avinashi Road", "Tiruppur North", "Tiruppur South", "PN Road", "Kangeyam Road", "Dharapuram Road", "Palladam Road"],
    "Mumbai": ["Andheri", "Bandra", "Borivali", "Dadar", "Powai", "Thane", "Colaba", "Goregaon", "Kurla", "Chembur", "Juhu", "Malad", "Kandivali", "Worli", "Lower Parel", "Mulund", "Vashi", "Nerul", "Panvel"],
    "Pune": ["Hinjawadi", "Kothrud", "Baner", "Viman Nagar", "Hadapsar", "Wakad", "Pimpri", "Aundh", "Kharadi", "Koregaon Park", "Shivajinagar", "Camp", "Warje", "Pashan", "Magarpatta", "Bhosari", "Dhayari"],
    "Navi Mumbai": ["Vashi", "Nerul", "Belapur", "Kharghar", "Airoli", "Ghansoli", "Koparkhairane", "Panvel", "Ulwe", "Turbhe"],
    "Nagpur": ["Dharampeth", "Sitabuldi", "Manish Nagar", "Sadar", "Civil Lines", "Wardha Road", "Hingna", "Nandanvan", "Mahal"],
    "Hyderabad": ["Madhapur", "Hitech City", "Gachibowli", "Kukatpally", "Banjara Hills", "Jubilee Hills", "Secunderabad", "Ameerpet", "Begumpet", "Miyapur", "Kondapur", "Manikonda", "LB Nagar", "Uppal", "Himayatnagar", "Mehdipatnam", "Financial District"],
    "Visakhapatnam": ["Madhurawada", "Gajuwaka", "Dwaraka Nagar", "Rushikonda", "Seethammadhara", "MVP Colony", "Beach Road", "NAD Junction", "Anakapalle", "Pendurthi"],
    "Vijayawada": ["Benz Circle", "Governorpet", "Moghalrajpuram", "Auto Nagar", "Bhavanipuram", "Patamata", "Poranki", "Kanuru", "Gunadala"],
    "Guntur": ["Brodipet", "Arundelpet", "Lakshmipuram", "Pattabhipuram", "Nallapadu", "Ring Road"],
    "Kolkata": ["Salt Lake", "New Town", "Park Street", "Behala", "Dum Dum", "Howrah", "Garia", "Ballygunge", "Alipore", "Tollygunge", "Jadavpur", "Esplanade", "Rajarhat", "Shyambazar", "Sealdah"],
    "Ahmedabad": ["Satellite", "Bopal", "Maninagar", "Navrangpura", "Prahlad Nagar", "Gota", "Thaltej", "Vastrapur", "Chandkheda", "Naranpura", "Motera", "SG Highway"],
    "Surat": ["Adajan", "Vesu", "Varachha", "Katargam", "Udhna", "Piplod", "City Light", "Pal", "Dumas Road", "Athwa"],
    "Vadodara": ["Alkapuri", "Manjalpur", "Gotri", "Akota", "Fatehgunj", "Sayajigunj", "Vasna", "Karelibaug", "Waghodia Road"],
    "Jaipur": ["Malviya Nagar", "Vaishali Nagar", "Mansarovar", "C-Scheme", "Jagatpura", "Raja Park", "Bani Park", "MI Road", "Jhotwara", "Sanganer"],
    "Jodhpur": ["Paota", "Sardarpura", "Ratanada", "Basni", "Shastri Nagar", "Mandore", "Chopasni Road"],
    "Udaipur": ["Hiran Magri", "Fatehpura", "City Centre", "Surajpole", "Pratap Nagar", "Bhopalpura", "Shobhagpura"],
    "Lucknow": ["Gomti Nagar", "Hazratganj", "Aliganj", "Indira Nagar", "Aminabad", "Chowk", "Mahanagar", "Vikas Nagar", "Alambagh", "Jankipuram", "Sushant Golf City"],
    "Kanpur": ["Civil Lines", "Swaroop Nagar", "Govind Nagar", "Kakadeo", "Kidwai Nagar", "Mall Road", "Kalyanpur", "Panki"],
    "Varanasi": ["Sigra", "Lanka", "Godowlia", "Bhelupur", "Assi", "Mahmoorganj", "Cantonment", "Rath Yatra"],
    "Agra": ["Tajganj", "Sikandra", "Dayal Bagh", "Kamla Nagar", "Sanjay Place", "Shahganj", "Fatehabad Road"],
    "Noida": ["Sector 18", "Sector 62", "Sector 50", "Sector 137", "Sector 15", "Greater Noida West", "Sector 76", "Sector 104", "Sector 125", "Sector 16", "Sector 63", "Sector 78"],
    "Ghaziabad": ["Indirapuram", "Vaishali", "Vasundhara", "Raj Nagar", "Crossings Republik", "Kaushambi", "Sahibabad", "Mohan Nagar"],
    "Gurugram": ["Cyber City", "Sector 29", "Golf Course Road", "Sohna Road", "Sector 56", "Udyog Vihar", "Sector 14", "Sector 10", "Sector 45", "Sector 49", "DLF Phase 1", "DLF Phase 3", "New Gurgaon"],
    "Faridabad": ["Sector 15", "Sector 16", "Sector 21", "NIT", "Old Faridabad", "Greater Faridabad", "Sector 37", "Sector 86"],
    "Ludhiana": ["Model Town", "Civil Lines", "Sarabha Nagar", "Ferozepur Road", "BRS Nagar", "Dugri", "Gill Road", "Pakhowal Road"],
    "Amritsar": ["Ranjit Avenue", "Lawrence Road", "Majitha Road", "Hall Bazaar", "Golden Temple Area", "Mall Road", "Airport Road"],
    "Patna": ["Boring Road", "Kankarbagh", "Rajendra Nagar", "Patliputra Colony", "Bailey Road", "Danapur", "Gandhi Maidan", "Fraser Road"],
    "Indore": ["Vijay Nagar", "Palasia", "Rajwada", "Bhawarkua", "Scheme 54", "AB Road", "Rau", "Sudama Nagar", "Mhow Naka"],
    "Bhopal": ["MP Nagar", "Arera Colony", "New Market", "Kolar Road", "Habibganj", "Bairagarh", "Shahpura", "TT Nagar"],
    "Bhubaneswar": ["Patia", "Jaydev Vihar", "Saheed Nagar", "Old Town", "Unit 4", "Khandagiri", "Nayapalli", "Rasulgarh", "Chandrasekharpur"],
    "Guwahati": ["Dispur", "Paltan Bazaar", "Beltola", "Six Mile", "Zoo Road", "Pan Bazaar", "Fancy Bazaar", "Khanapara", "Maligaon"],
    "Ranchi": ["Lalpur", "Harmu", "Morabadi", "Doranda", "Main Road", "Bariatu", "Kanke", "Namkum"],
    "Raipur": ["Shankar Nagar", "Pandri", "Telibandha", "Civil Lines", "Devendra Nagar", "Amanaka", "Mowa", "Tatibandh"],
    "Dehradun": ["Rajpur Road", "ISBT", "Clock Tower", "Vasant Vihar", "Ballupur", "Prem Nagar", "Dalanwala", "Sahastradhara Road"],
    "Shimla": ["Mall Road", "Sanjauli", "New Shimla", "Summer Hill", "Boileauganj", "Khalini", "Tara Devi"],
    "Srinagar": ["Lal Chowk", "Rajbagh", "Hazratbal", "Bemina", "Jawahar Nagar", "Hyderpora", "Nishat", "Nowgam"],
    "Jammu": ["Gandhi Nagar", "Trikuta Nagar", "Jewel Chowk", "Bahu Plaza", "Janipur", "Channi Himmat", "Talab Tillo"],
    "Shillong": ["Police Bazaar", "Laitumkhrah", "Nongthymmai", "Mawlai", "Polo", "Upper Shillong"],
    "Agartala": ["Agartala City", "Pratapgarh", "Bardowali", "Banamalipur", "Battala", "Dhaleswar"],
    "Panaji": ["Miramar", "Fontainhas", "Taleigao", "Panjim Market", "Caranzalem", "Dona Paula"],
    "Margao": ["Fatorda", "Borda", "Colva Road", "Margao Market", "Navelim"],
    "Chandigarh": ["Sector 17", "Sector 22", "Sector 35", "Manimajra", "Sector 8", "Sector 9", "Sector 26", "Sector 43", "Sector 45", "Sector  Panchkula Road"],
    "Puducherry": ["White Town", "Lawspet", "Ariyankuppam", "Muthialpet", "Reddiarpalayam", "Villianur", "Oulgaret", "Anna Nagar"],
    "Port Blair": ["Aberdeen", "Phoenix Bay", "Garacharma", "Dollygunj", "Junglighat", "Bathubasti"],
    "Daman": ["Nani Daman", "Moti Daman", "Vapi Road", "Dunetha"],
    "Silvassa": ["Silvassa Town", "Vapi Road", "Amli", "Khanvel Road"],
    "Leh": ["Leh Town", "Choglamsar", "Skara", "Changspa", "Saboo"],
    "Kavaratti": ["Kavaratti Town", "Beach Area", "North Kavaratti"],
})

with st.sidebar:
    st.header("Your business plan")
    state = st.selectbox("State / Union Territory", STATES, index=STATES.index("Kerala"), key="state_dropdown")
    available_cities = CITY_LISTS.get(state, [])
    if not available_cities:
        st.warning("City dropdown list is not configured for this state/UT yet. Please choose another state or add cities to CITY_LISTS in the code.")
        city = ""
        locality = ""
    else:
        default_city_index = available_cities.index("Kochi") if state == "Kerala" and "Kochi" in available_cities else 0
        city = st.selectbox("City / town", available_cities, index=default_city_index, key=f"city_dropdown_{state}", help="Choose from the predefined city list.")
        available_localities = NEIGHBOURHOOD_LISTS.get(city, [])
        if available_localities:
            locality = st.selectbox("Neighbourhood / locality", available_localities, key=f"locality_dropdown_{city}", help="Choose a neighbourhood available for the selected city.")
        else:
            locality = ""
            st.warning(f"Neighbourhood dropdown is not configured for {city} yet. Add its localities to NEIGHBOURHOOD_LISTS in the code.")
    radius_km = st.selectbox("Market search radius (km)", [0.5,1.0,1.5,2.0,3.0,5.0], index=2)
    budget = st.number_input("Fixed investment budget (₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
    rent = st.number_input("Expected monthly shop rent (₹)", min_value=0, max_value=10000000, value=15000, step=1000)
    population = st.number_input("Local population estimate (if known)", min_value=0, max_value=100000000, value=10000, step=1000, help="User-entered estimate; not automatically verified.")
    categories = st.multiselect("Shop categories to compare", list(PROFILES), default=list(PROFILES))
    priority = st.selectbox("Recommendation priority", ["Balanced (recommended)","Demand priority","Profit priority","Low competition priority","Budget-fit priority"], help="Relative ranking weights, not a success probability.")
    analyse = st.button("Analyse this location", type="primary", use_container_width=True)
    retry = st.button("Retry / refresh map data", use_container_width=True)
    st.divider()
    st.subheader("Add local verified datasets (CSV)")
    st.caption("Upload geocoded CSV files. Facility/business files need latitude and longitude columns. Survey files are read separately and are not counted as mapped places.")
    gov_csv = st.file_uploader("Government / MSME businesses", type=["csv"], key="gov_csv")
    directory_csv = st.file_uploader("Business directory", type=["csv"], key="directory_csv")
    schools_csv = st.file_uploader("Schools / colleges", type=["csv"], key="schools_csv")
    hospitals_csv = st.file_uploader("Hospitals / clinics", type=["csv"], key="hospitals_csv")
    survey_csv = st.file_uploader("Customer survey", type=["csv"], key="survey_csv")

if retry:
    geocode_location.clear()
    st.info("Geocoding and live map cache cleared. Click Analyse this location to retry.")

model, model_error = load_model()
if model is not None:
    st.caption("Saved Random Forest loaded. ML is used only if its original input features are compatible and mapped counts are available; otherwise labelled planning fallback is used.")
else:
    st.warning("Saved Random Forest could not be loaded. Rule-based fallback will be used.")
    if model_error: st.caption(model_error)

if analyse:
    if not city.strip() or not locality.strip(): st.error("Enter both city/town and locality."); st.stop()
    if not categories: st.error("Select at least one shop category."); st.stop()
    with st.spinner("Geocoding locality (POI results are rejected)..."):
        location = geocode_location(locality.strip(), city.strip(), state)
    if location is None:
        st.error("Could not locate this neighbourhood. We also tried city-level fallback, but could not identify the city coordinates. No business data request was made. Check the selected city/locality or try again later.")
        st.stop()
    st.subheader(f"Results for {locality.strip()}, {city.strip()}, {state}")
    st.caption(f"Location match: {location['display_name']} | Type: {location.get('place_type','unknown')} | Source: {location.get('source','unknown')}")
    if location.get("source") == "Configured locality fallback":
        st.warning("Using a configured fallback point for this locality. This point is not a verified locality boundary or centroid.")
    elif location.get("source") == "City-level fallback (neighbourhood match unavailable)":
        st.warning("The selected neighbourhood could not be geocoded. Analysis will search around the city-level point instead, so nearby-place counts may not represent the selected neighbourhood accurately.")
    radius_m = int(radius_km*1000)
    data_source = "Cached OpenStreetMap data"
    cache_message = None
    live_error = None
    live_server = None
    places = None
    uploaded_places, uploaded_layers, upload_notes, survey_rows = uploaded_csv_places(
        [("Government / MSME businesses", gov_csv), ("Business directory", directory_csv),
         ("Schools / colleges", schools_csv), ("Hospitals / clinics", hospitals_csv),
         ("Customer survey", survey_csv)],
        location["lat"], location["lon"], radius_m, locality.strip(), city.strip(), state)

    # Cache first for a quick result, then make a short live request only when needed.
    if not retry:
        with st.spinner("Checking saved local OSM data..."):
            places, cache_message = local_cache_places(location["lat"], location["lon"], radius_m, locality.strip(), city.strip(), state)
    if places is not None and not retry:
        data_source = "Cached OpenStreetMap data"
        st.success(f"Using saved OSM data: {len(places):,} records.")
        st.caption(cache_message or "Saved data found.")
    else:
        with st.spinner("Trying live map data (short timeout; cache and uploaded CSVs remain available)..."):
            live_places, live_error, live_server = live_osm_places(location["lat"], location["lon"], radius_m)
        if live_places is not None:
            places = live_places
            data_source = "Live OpenStreetMap data"
            st.success(f"Live OSM returned {len(live_places):,} mapped records.")
            try:
                cache_total = save_live_cache(live_places, locality.strip(), city.strip(), state, radius_m, OSM_LAYER_STATUS)
                st.caption(f"Live data saved in cache: {CACHE_V2} | Cache rows: {cache_total:,}")
            except Exception as exc:
                st.caption(f"Live data retrieved, but cache save failed: {exc}")
        else:
            places = None
            data_source = "Uploaded local CSV data" if uploaded_places else "Assumption-only fallback"
            st.warning("Live OSM is unavailable. The app will use any uploaded geocoded CSV records; missing layers will remain unknown.")
            with st.expander("Data retrieval details"):
                st.write(live_error or "No live API error details.")
                st.write(cache_message or "No matching cache details.")

    # Combine uploaded data with any available OSM/cache records.
    osm_data_available = places is not None
    osm_places = places or []
    places = merge_place_records(osm_places, uploaded_places)
    base_status = OSM_LAYER_STATUS.copy() if osm_data_available else {"shops": None, "food": None, "education": None, "health": None}
    for layer, has_rows in uploaded_layers.items():
        if has_rows:
            base_status[layer] = True
    OSM_LAYER_STATUS = base_status
    if uploaded_places:
        data_source += " + uploaded CSV records" if osm_places else ""
        st.info(f"Uploaded geocoded records matched to this radius: {len(uploaded_places):,}. These are records in your files, not guaranteed complete totals.")
    if upload_notes:
        with st.expander("Uploaded dataset details"):
            for note in upload_notes:
                st.write("• " + note)
    if survey_rows:
        st.caption(f"Customer survey CSV: {survey_rows:,} rows read for this run. Responses are separate from mapped competitor/facility counts and are not yet used in demand calculations or model training.")
    if not places and not osm_data_available:
        places = None
    with st.expander("Data coverage by information type", expanded=True):
        labels = {"shops":"Competitor shops and services", "food":"Cafes, restaurants and bakeries", "education":"Schools and colleges", "health":"Hospitals, clinics and pharmacies"}
        for key, label in labels.items():
            state_value = OSM_LAYER_STATUS.get(key)
            if state_value is True: st.write(f"✅ {label}: records available from a completed OSM query and/or uploaded CSV")
            elif state_value is False: st.write(f"⚠️ {label}: live OSM request failed; no uploaded records matched")
            else: st.write(f"ℹ️ {label}: no verified records available; count remains unknown")
        if live_error: st.caption(live_error)
    result = make_comparison(categories, model, locality.strip(), int(population), places, float(rent), float(budget))
    result = add_suitability_scores(result, places is not None, priority)
    eligible = result[result["Within budget"] == "Yes"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Investment budget", money(budget)); c2.metric("Categories compared",len(result)); c3.metric("Options within budget",len(eligible))
    c4.metric("Highest estimated monthly profit", money(eligible.iloc[0]["Estimated monthly operating profit (₹)"]) if not eligible.empty else "None in budget")
    if not eligible.empty:
        best=eligible.iloc[0]
        st.subheader("🏆 Best shop recommendation")
        st.success(f"**{best['Shop category']}** — relative suitability score **{best['Suitability Score']:.1f}/100** ({priority}).")
        a,b,c=st.columns(3); a.metric("Estimated monthly profit",money(best["Estimated monthly operating profit (₹)"])); b.metric("Estimated monthly demand",f"{int(best['Estimated monthly transactions']):,} transactions"); c.metric("Estimated upfront investment",money(best["Estimated upfront investment (₹)"]))
        st.markdown("**Why ranked first**")
        st.write(f"• Profit score: {best['Profit Score']:.1f}/100 (relative to chosen categories)")
        st.write(f"• Demand score: {best['Demand Score']:.1f}/100 (estimated, not observed demand)")
        st.write(f"• Budget fit: {best['Budget Score']:.1f}/100")
        if places is not None: st.write(f"• Competition score: {best['Competition Score']:.1f}/100 based on mapped records; OSM may be incomplete")
        else: st.write("• Competition unavailable; neutral score used, not zero competitors")
        if places is None: st.warning("Assumption-based shortlist only; validate the market locally before investing.")
        elif best["Estimated monthly operating profit (₹)"] <= 0: st.warning("Top eligible option has estimated zero/negative operating profit. Ranking is not a positive-profit guarantee.")
    else:
        st.warning("No selected category fits the budget based on assumed setup, stock and four months of rent.")
    st.subheader("Shop category comparison")
    st.caption("Counts are mapped records from available OSM/cache and uploaded geocoded CSV files within the selected radius. Uploaded files may be incomplete and are not an official census. A zero from a completed OSM query means no matching OSM records were returned; it does not prove no facility exists. Failed or unverified layers remain unknown. None means no usable records were available for that layer.")
    st.caption("Suitability score is a relative decision score, not a probability of business success.")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.subheader("Estimated monthly operating profit")
    import matplotlib.pyplot as plt
    chart = result[["Shop category","Estimated monthly operating profit (₹)"]].sort_values("Estimated monthly operating profit (₹)")
    fig,ax=plt.subplots(figsize=(10,max(4,.48*len(chart))))
    ax.barh(chart["Shop category"],chart["Estimated monthly operating profit (₹)"]); ax.axvline(0,linewidth=1)
    ax.set_xlabel("Estimated monthly operating profit (₹)"); ax.set_ylabel("Shop category"); ax.set_title("Estimated profit/loss (assumption-based)"); ax.grid(axis="x",alpha=.25)
    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.subheader("Calculation notes")
    st.markdown(f"- **Data source:** {data_source}. Unknown mapped counts are not treated as zero.\n- **Revenue:** estimated transactions × assumed average bill.\n- **Operating cost:** COGS + staff + utilities + marketing + rent ({money(rent)}).\n- **Break-even:** approximate transactions needed to cover fixed costs.\n- **Upfront investment:** setup + opening stock + 4 months of entered rent (planning assumption).\n- **Payback:** upfront cost divided by estimated monthly profit when profit is positive.\n- **Population:** user-provided, not verified automatically.\n- **Demand method:** Random Forest estimate if compatible; otherwise rule-based planning estimate.")
    st.download_button("Download comparison CSV", data=result.to_csv(index=False).encode("utf-8-sig"), file_name="india_shop_business_comparison.csv", mime="text/csv")
    st.caption("Planning aid only. Validate footfall, actual competitors, rent, suppliers, licences and customer demand locally.")
else:
    st.subheader("How to use")
    st.markdown("1. Select State/UT from the dropdown.\n2. Select a city from the predefined city list.\n3. Select a neighbourhood from the city-specific dropdown.\n4. Enter budget and expected monthly rent.\n5. Enter population estimate if known, choose shop categories, and click **Analyse this location**.")
    st.warning("Live OSM data availability varies by locality. Use verified population, actual rent quotations and customer survey information for better decisions.")
