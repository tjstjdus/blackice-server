import os
import glob
import joblib
import requests
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.neighbors import BallTree

# =========================================================
# FastAPI
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 시간
# =========================================================

KST = timezone(timedelta(hours=9))

def now_kst():
    """서버 위치(타임존)와 무관하게 항상 한국 표준시 기준 현재 시각 반환"""
    return datetime.now(KST).replace(tzinfo=None)

# =========================================================
# BASE DIR
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# API KEY
# =========================================================

KMA_API_KEY = os.getenv(
    "KMA_API_KEY",
    "여기에_API_KEY"
)

# =========================================================
# 기상청 API
# =========================================================

PAST_URL = (
    "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
)

# 단기예보 (현재 ~ 미래 최대 3일, 1시간 간격) — 격자(nx, ny) 기반
FORECAST_URL = (
    "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
)

# 초단기실황 (방금 관측된 가장 최신 값, 매시 10분에 갱신) — 격자(nx, ny) 기반
NCST_URL = (
    "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtNcst"
)

# =========================================================
# 파일 찾기
# =========================================================

def find_file(patterns):

    for pattern in patterns:

        files = glob.glob(
            os.path.join(BASE_DIR, pattern)
        )

        files += glob.glob(
            os.path.join(BASE_DIR, "data", pattern)
        )

        files += glob.glob(
            os.path.join(BASE_DIR, "models", pattern)
        )

        if files:
            return files[0]

    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: {patterns}"
    )

DATA_PATH = find_file([
    "결빙_비결빙_전국데이터*.csv"
])

META_PATH = find_file([
    "META_관측지점정보*.csv"
])

ICING_MODEL_PATH = find_file([
    "결빙확률모델*.pkl"
])

BLACKICE_MODEL_PATH = find_file([
    "블랙아이스확률모델*.pkl"
])

# =========================================================
# CSV 읽기
# =========================================================

def read_csv_safe(path):

    for enc in [
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "utf-8"
    ]:

        try:

            return pd.read_csv(
                path,
                encoding=enc,
                low_memory=False
            )

        except UnicodeDecodeError:
            continue

    raise Exception(
        f"CSV 인코딩 읽기 실패: {path}"
    )

# =========================================================
# 데이터 로드
# =========================================================

base_df = read_csv_safe(DATA_PATH)

meta_df = read_csv_safe(META_PATH)

# =========================================================
# 최근접 ASOS 연결
# =========================================================

def attach_nearest_asos(
    base_df,
    meta_df
):

    base_df = base_df.copy()
    meta_df = meta_df.copy()

    meta_df = meta_df.rename(
        columns={
            "지점": "asos_id",
            "지점명": "asos_name"
        }
    )

    base_df["위도"] = pd.to_numeric(
        base_df["위도"],
        errors="coerce"
    )

    base_df["경도"] = pd.to_numeric(
        base_df["경도"],
        errors="coerce"
    )

    meta_df["asos_id"] = pd.to_numeric(
        meta_df["asos_id"],
        errors="coerce"
    )

    meta_df["asos_lat"] = pd.to_numeric(
        meta_df["위도"],
        errors="coerce"
    )

    meta_df["asos_lon"] = pd.to_numeric(
        meta_df["경도"],
        errors="coerce"
    )

    base_df = base_df.dropna(
        subset=["위도", "경도"]
    ).reset_index(drop=True)

    meta_df = meta_df.dropna(
        subset=[
            "asos_id",
            "asos_lat",
            "asos_lon"
        ]
    ).reset_index(drop=True)

    base_rad = np.radians(
        base_df[["위도", "경도"]].values
    )

    meta_rad = np.radians(
        meta_df[["asos_lat", "asos_lon"]].values
    )

    tree = BallTree(
        meta_rad,
        metric="haversine"
    )

    dist, idx = tree.query(
        base_rad,
        k=1
    )

    earth_radius_km = 6371.0088

    matched = meta_df.iloc[
        idx[:, 0]
    ].reset_index(drop=True)

    base_df["asos_id"] = matched[
        "asos_id"
    ].astype(int).astype(str)

    base_df["asos_name"] = matched[
        "asos_name"
    ].values

    base_df["asos_distance_m"] = (
        dist[:, 0]
        * earth_radius_km
        * 1000
    )

    return base_df

base_df = attach_nearest_asos(
    base_df,
    meta_df
)

print(f"[DATA] 로드된 CSV 파일: {DATA_PATH}")
print(f"[DATA] 전체 지점 수: {len(base_df)}행")

# =========================================================
# 모델 로드
# =========================================================

def get_model_and_features(obj):

    if isinstance(obj, dict):

        model = obj["model"]

        if "features" in obj:

            features = list(
                obj["features"]
            )

        elif hasattr(
            model,
            "feature_names_in_"
        ):

            features = list(
                model.feature_names_in_
            )

        else:

            features = []

        return model, features

    model = obj

    if hasattr(
        model,
        "feature_names_in_"
    ):

        features = list(
            model.feature_names_in_
        )

    else:

        features = []

    return model, features

icing_obj = joblib.load(
    ICING_MODEL_PATH
)

blackice_obj = joblib.load(
    BLACKICE_MODEL_PATH
)

icing_model, icing_features = \
    get_model_and_features(
        icing_obj
    )

blackice_model, blackice_features = \
    get_model_and_features(
        blackice_obj
    )

# =========================================================
# 지역 목록
# =========================================================

regions = {}

for province in sorted(
    base_df["시도"].dropna().unique()
):

    cities = sorted(
        base_df[
            base_df["시도"] == province
        ]["시군구"].dropna().unique()
    )

    regions[province] = cities

# =========================================================
# Request
# =========================================================

class PredictRequest(BaseModel):

    date: str
    time: str

    province: str
    city: str

    max_points: int = 20

# =========================================================
# 추정노면온도 계산
# 노트북(예측모델_실시간_수정.ipynb)과 동일한 로직.
# 기본 가중합 공식에 더해, 여름/고온 조건에서 비현실적으로
# 낮은 노면온도가 나오지 않도록 3단계 보정을 적용한다.
# =========================================================

def add_estimated_road_surface_temp(df):

    df = df.copy()

    for col in ["기온", "풍속", "강수량", "지면온도"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 지면온도 결측이면 기온으로 대체
    df["지면온도"] = df["지면온도"].fillna(df["기온"])
    df["풍속"] = df["풍속"].fillna(0)
    df["강수량"] = df["강수량"].fillna(0)

    # 1) 지면온도가 비정상적으로 낮으면 기온으로 보정
    #    (예: 기온은 영상인데 지면온도만 결측 보간 등으로 비정상적으로 낮게 잡힌 경우)
    df.loc[
        (df["기온"] >= 10) & (df["지면온도"] < df["기온"] - 10),
        "지면온도"
    ] = df["기온"]

    # 기본 추정노면온도 계산
    df["추정노면온도"] = (
        0.7 * df["기온"]
        + 0.2 * df["지면온도"]
        - 0.3 * df["풍속"]
        - 0.1 * df["강수량"]
    )

    # 2) 여름/고온 조건에서 비정상적 저온 방지
    df.loc[
        (df["기온"] >= 10) & (df["추정노면온도"] < df["기온"] - 5),
        "추정노면온도"
    ] = df["기온"]

    # 3) 기온이 5도 이상이면 추정노면온도가 음수로 내려가지 않게 제한
    df.loc[
        (df["기온"] >= 5) & (df["추정노면온도"] < 0),
        "추정노면온도"
    ] = df["기온"]

    return df

# =========================================================
# 위험도
# =========================================================

def make_risk_level(prob):

    if prob >= 0.8:
        return "매우 위험"

    elif prob >= 0.6:
        return "위험"

    elif prob >= 0.3:
        return "주의"

    else:
        return "낮음"

# =========================================================
# 모델 입력 생성
# =========================================================

def make_model_input(
    df,
    feature_cols
):

    X = pd.DataFrame()

    for col in feature_cols:

        if col in df.columns:

            X[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

        else:

            X[col] = 0

    return X

# =========================================================
# 결측 기상값 보간 — 가까운 다른 지점(관측소)의 값으로 대체
# 0℃, 70% 같은 임의 기본값 대신, 실제로 관측된 가장 가까운
# 지점의 값을 사용하여 비현실적인 예측(예: 여름에 위험도 급등)을 방지
# =========================================================

def fill_missing_with_nearest(df, target_cols, lat_col="위도", lon_col="경도"):

    df = df.copy()

    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan

    valid_mask = df[target_cols].notna().any(axis=1)

    has_valid = df[valid_mask]
    missing = df[~valid_mask]

    if missing.empty or has_valid.empty:
        return df

    valid_coords = np.radians(
        has_valid[[lat_col, lon_col]].astype(float).values
    )
    missing_coords = np.radians(
        missing[[lat_col, lon_col]].astype(float).values
    )

    tree = BallTree(valid_coords, metric="haversine")
    _, idx = tree.query(missing_coords, k=1)

    nearest_values = has_valid.iloc[idx[:, 0]][target_cols].reset_index(drop=True)

    missing_filled = missing.reset_index()
    for col in target_cols:
        missing_filled[col] = nearest_values[col].values
    missing_filled = missing_filled.set_index("index")

    df.update(missing_filled)

    return df

# =========================================================
# 도로 스내핑 — 좌표를 가장 가까운 실제 도로 위로 보정
# OSRM(Open Source Routing Machine) 공개 API 사용, 별도 키 불필요
#
# 전국 지점을 매 요청마다 스내핑하면 시간이 오래 걸리므로(전체 약 1700개
# 기준 병렬 처리해도 약 1분 소요), 서버 시작 전에 1회성 스크립트
# (scripts/build_snap_cache.py)로 미리 계산해 캐시 CSV로 저장해두고,
# 서버는 그 캐시를 읽기만 한다.
# =========================================================

OSRM_NEAREST_URL = "https://router.project-osrm.org/nearest/v1/driving"

SNAP_CACHE_PATH = os.path.join(BASE_DIR, "snap_cache.csv")

def snap_one_point(lat, lon):
    """단일 좌표를 가장 가까운 도로 위 좌표로 보정. 실패 시 원본 좌표 반환.
    (사전 캐시 생성 스크립트에서만 사용 — 서버 요청 처리 중에는 호출하지 않음)
    """

    try:
        url = f"{OSRM_NEAREST_URL}/{lon},{lat}"
        response = requests.get(url, params={"number": 1}, timeout=5)
        response.raise_for_status()
        data = response.json()

        waypoints = data.get("waypoints", [])
        if not waypoints:
            return (lat, lon)

        snapped_lon, snapped_lat = waypoints[0]["location"]

        # 보정 거리가 비정상적으로 크면(예: 매칭 실패) 원본 좌표 사용
        dist_m = waypoints[0].get("distance", 0)
        if dist_m is not None and dist_m > 500:
            return (lat, lon)

        return (float(snapped_lat), float(snapped_lon))

    except Exception as e:
        print(f"도로 스내핑 실패 ({lat},{lon}):", str(e))
        return (lat, lon)

def build_snap_cache(df, lat_col="위도", lon_col="경도", max_workers=10):
    """
    df의 모든 unique 좌표를 도로 위로 보정하여
    [원본위도, 원본경도, snapped_lat, snapped_lon] CSV를 생성한다.

    1회성 사전 처리용 함수 — 서버 기동 시가 아니라
    `python main.py --build-snap-cache` 같은 별도 실행에서만 호출한다.
    """

    unique_coords = df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    coord_list = list(zip(
        unique_coords[lat_col].astype(float),
        unique_coords[lon_col].astype(float)
    ))

    print(f"도로 스내핑 캐시 생성 시작 — 총 {len(coord_list)}개 좌표")

    rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(snap_one_point, lat, lon): (lat, lon)
            for lat, lon in coord_list
        }

        done = 0
        for future in as_completed(futures):
            orig_lat, orig_lon = futures[future]
            try:
                snapped_lat, snapped_lon = future.result()
            except Exception:
                snapped_lat, snapped_lon = orig_lat, orig_lon

            rows.append({
                "원위도": orig_lat,
                "원경도": orig_lon,
                "snapped_lat": snapped_lat,
                "snapped_lon": snapped_lon
            })

            done += 1
            if done % 100 == 0:
                print(f"  진행: {done}/{len(coord_list)}")

    cache_df = pd.DataFrame(rows)
    cache_df.to_csv(SNAP_CACHE_PATH, index=False, encoding="utf-8-sig")

    print(f"도로 스내핑 캐시 저장 완료: {SNAP_CACHE_PATH} ({len(cache_df)}개 좌표)")

    return cache_df

def load_snap_cache():
    """서버 기동 시 캐시 CSV를 로드. 없으면 빈 매핑(보정 없음)으로 동작."""

    if not os.path.exists(SNAP_CACHE_PATH):
        print(f"[경고] 도로 스내핑 캐시 파일이 없습니다: {SNAP_CACHE_PATH}")
        print("       원본 좌표를 그대로 사용합니다.")
        print("       caches는 'python main.py --build-snap-cache' 로 미리 생성하세요.")
        return {}

    cache_df = read_csv_safe(SNAP_CACHE_PATH)

    mapping = {}
    for _, row in cache_df.iterrows():
        key = (round(float(row["원위도"]), 6), round(float(row["원경도"]), 6))
        mapping[key] = (float(row["snapped_lat"]), float(row["snapped_lon"]))

    print(f"도로 스내핑 캐시 로드 완료: {len(mapping)}개 좌표")
    return mapping

# 서버 기동 시 1회 로드 — 이후 모든 요청은 이 메모리 캐시를 조회만 함
SNAP_CACHE = load_snap_cache()

def apply_snapped_coords(df, lat_col="위도", lon_col="경도"):
    """
    캐시에 보정 좌표가 있으면 위도/경도 컬럼을 보정값으로 덮어쓴다.
    캐시에 없는 좌표는 원본 좌표를 그대로 유지한다.
    (OSRM 호출 없음 — 메모리 딕셔너리 조회만 수행)
    """

    df = df.copy()

    def get_snapped(row):
        key = (round(float(row[lat_col]), 6), round(float(row[lon_col]), 6))
        if key in SNAP_CACHE:
            return SNAP_CACHE[key]
        return (row[lat_col], row[lon_col])

    snapped = df.apply(get_snapped, axis=1)
    df[lat_col] = snapped.apply(lambda x: x[0])
    df[lon_col] = snapped.apply(lambda x: x[1])

    return df

# =========================================================
# JSON 정리
# =========================================================

def clean_json_value(value):

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (np.float32, np.float64)
    ):
        return float(value)

    if isinstance(
        value,
        (np.int32, np.int64)
    ):
        return int(value)

    return value

def dataframe_to_json_records(df):

    records = df.to_dict(
        orient="records"
    )

    clean_records = []

    for row in records:

        clean_row = {}

        for key, value in row.items():

            clean_row[key] = \
                clean_json_value(value)

        clean_records.append(
            clean_row
        )

    return clean_records

# =========================================================
# safe float
# =========================================================

def safe_float(x):

    try:

        value = float(x)

        if value in [
            -9,
            -9.0,
            -99,
            -99.0
        ]:
            return np.nan

        return value

    except:
        return np.nan

# =========================================================
# 기상 데이터
# =========================================================

def fetch_weather_data(target_time):
    tm = target_time.strftime("%Y%m%d%H00")
    params = {"tm": tm, "stn": "0", "help": "0", "authKey": KMA_API_KEY}

    try:
        response = requests.get(PAST_URL, params=params, timeout=30)
        response.raise_for_status()

        lines = response.text.split("\n")
        weather_list = []

        for line in lines:
            if line.startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.split()
            if len(parts) < 39:
                continue

            try:
                weather_list.append({
                    "asos_id":  str(parts[1]),
                    "기온":      safe_float(parts[11]),
                    "습도":      safe_float(parts[13]),
                    "풍향":      safe_float(parts[2]),
                    "풍속":      safe_float(parts[3]),
                    "강수량":    safe_float(parts[15]),
                    "지면온도":  safe_float(parts[36])
                })
            except:
                continue

        return pd.DataFrame(weather_list)  # ✅ rows → weather_list

    except Exception as e:
        print("기상청 API 오류:", str(e))
        return pd.DataFrame()

# =========================================================
# 위경도 → 기상청 격자좌표(nx, ny) 변환
# 기상청 단기예보 API는 5km 격자(Lambert Conformal Conic) 좌표를 씀
# =========================================================

def latlon_to_grid(lat, lon):

    RE = 6371.00877       # 지구 반경(km)
    GRID = 5.0             # 격자 간격(km)
    SLAT1 = 30.0           # 표준위도1
    SLAT2 = 60.0           # 표준위도2
    OLON = 126.0           # 기준점 경도
    OLAT = 38.0            # 기준점 위도
    XO = 43                # 기준점 X좌표
    YO = 136                # 기준점 Y좌표

    DEGRAD = np.pi / 180.0

    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = np.tan(np.pi * 0.25 + slat2 * 0.5) / np.tan(np.pi * 0.25 + slat1 * 0.5)
    sn = np.log(np.cos(slat1) / np.cos(slat2)) / np.log(sn)

    sf = np.tan(np.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * np.cos(slat1) / sn

    ro = np.tan(np.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)

    ra = np.tan(np.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)

    theta = lon * DEGRAD - olon
    if theta > np.pi:
        theta -= 2.0 * np.pi
    if theta < -np.pi:
        theta += 2.0 * np.pi
    theta *= sn

    nx = int(ra * np.sin(theta) + XO + 0.5)
    ny = int(ro - ra * np.cos(theta) + YO + 0.5)

    return nx, ny

# =========================================================
# 단기예보 base_time 계산
# 단기예보는 02,05,08,11,14,17,20,23시에 발표되며,
# 발표 직후 일정 시간(약 10분) 데이터 정리 시간이 필요함
# =========================================================

def get_forecast_base(now):

    base_hours = [2, 5, 8, 11, 14, 17, 20, 23]

    candidate = now - timedelta(minutes=10)

    valid_hours = [h for h in base_hours if h <= candidate.hour]

    if valid_hours:
        base_hour = max(valid_hours)
        base_date = candidate.strftime("%Y%m%d")
    else:
        # 자정 직후 — 전날 23시 발표본 사용
        prev_day = candidate - timedelta(days=1)
        base_hour = 23
        base_date = prev_day.strftime("%Y%m%d")

    base_time = f"{base_hour:02d}00"

    return base_date, base_time

# =========================================================
# 초단기실황 base_time 계산
# 초단기실황은 매시 정각 발표, 약 10분 후 데이터 제공
# 예: 14:05 요청 → 아직 14시 데이터 없음 → 13시 데이터 사용
#     14:15 요청 → 14시 데이터 사용 가능
# =========================================================

def get_ncst_base(now):

    candidate = now - timedelta(minutes=10)

    base_date = candidate.strftime("%Y%m%d")
    base_time = candidate.strftime("%H00")

    return base_date, base_time

# =========================================================
# 단기예보 데이터 (미래 시각 — 최대 약 3일 후까지)
# nx, ny 격자별로 1회씩 요청하므로, 호출 전 좌표를 중복 제거해서 넘길 것
# =========================================================

def fetch_forecast_data(target_time, grid_points):
    """
    target_time : 예보를 보고 싶은 미래 시각 (datetime)
    grid_points : [(asos_id, lat, lon), ...] 형태의 리스트
                  (asos_id별 대표 좌표 — 보통 ASOS 위치나 지점 위경도)

    격자(nx, ny)별로 병렬 호출하여 응답 시간을 단축한다.
    (전국 지점 호출 시 순차 처리하면 타임아웃/CORS 오류로 이어질 수 있음)
    """

    base_date, base_time = get_forecast_base(now_kst())

    fcst_date = target_time.strftime("%Y%m%d")
    fcst_hour = target_time.strftime("%H00")

    # nx, ny 기준으로 중복 제거 (같은 격자는 한 번만 호출)
    grid_map = {}
    for asos_id, lat, lon in grid_points:
        nx, ny = latlon_to_grid(lat, lon)
        grid_map.setdefault((nx, ny), []).append(asos_id)

    def fetch_one_grid(nx, ny, asos_ids):

        params = {
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
            "authKey": KMA_API_KEY
        }

        try:
            response = requests.get(FORECAST_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            items = (
                data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
            )

            point_data = {}

            for item in items:
                if item.get("fcstDate") != fcst_date:
                    continue
                if item.get("fcstTime") != fcst_hour:
                    continue

                category = item.get("category")
                value = item.get("fcstValue")

                if category == "TMP":
                    point_data["기온"] = safe_float(value)
                elif category == "REH":
                    point_data["습도"] = safe_float(value)
                elif category == "VEC":
                    point_data["풍향"] = safe_float(value)
                elif category == "WSD":
                    point_data["풍속"] = safe_float(value)
                elif category == "PCP":
                    point_data["강수량"] = safe_float(value) if value not in (
                        "강수없음", None
                    ) else 0.0

            if not point_data:
                return []

            point_data.setdefault("지면온도", point_data.get("기온"))
            point_data.setdefault("강수량", 0.0)

            rows = []
            for asos_id in asos_ids:
                row = {"asos_id": str(asos_id)}
                row.update(point_data)
                rows.append(row)
            return rows

        except Exception as e:
            print(f"단기예보 API 오류 (nx={nx}, ny={ny}):", str(e))
            return []

    weather_list = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(fetch_one_grid, nx, ny, asos_ids): (nx, ny)
            for (nx, ny), asos_ids in grid_map.items()
        }

        for future in as_completed(futures):
            weather_list.extend(future.result())

    return pd.DataFrame(weather_list)

# =========================================================
# 초단기실황 데이터 (현재 시각 — 가장 최신 관측값)
# =========================================================

def fetch_ncst_data(grid_points):
    """
    grid_points : [(asos_id, lat, lon), ...]
    "현재"는 시시각각 변하므로 target_time을 받지 않고
    항상 now_kst() 기준 최신 발표분을 사용

    격자(nx, ny)별로 병렬 호출하여 응답 시간을 단축한다.
    """

    base_date, base_time = get_ncst_base(now_kst())

    # nx, ny 기준으로 중복 제거 (같은 격자는 한 번만 호출)
    grid_map = {}
    for asos_id, lat, lon in grid_points:
        nx, ny = latlon_to_grid(lat, lon)
        grid_map.setdefault((nx, ny), []).append(asos_id)

    def fetch_one_grid(nx, ny, asos_ids):

        params = {
            "pageNo": 1,
            "numOfRows": 100,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
            "authKey": KMA_API_KEY
        }

        try:
            response = requests.get(NCST_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            items = (
                data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
            )

            point_data = {}

            for item in items:
                category = item.get("category")
                value = item.get("obsrValue")

                if category == "T1H":
                    point_data["기온"] = safe_float(value)
                elif category == "REH":
                    point_data["습도"] = safe_float(value)
                elif category == "VEC":
                    point_data["풍향"] = safe_float(value)
                elif category == "WSD":
                    point_data["풍속"] = safe_float(value)
                elif category == "RN1":
                    point_data["강수량"] = safe_float(value)

            if not point_data:
                return []

            point_data.setdefault("지면온도", point_data.get("기온"))
            point_data.setdefault("강수량", 0.0)

            rows = []
            for asos_id in asos_ids:
                row = {"asos_id": str(asos_id)}
                row.update(point_data)
                rows.append(row)
            return rows

        except Exception as e:
            print(f"초단기실황 API 오류 (nx={nx}, ny={ny}):", str(e))
            return []

    weather_list = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(fetch_one_grid, nx, ny, asos_ids): (nx, ny)
            for (nx, ny), asos_ids in grid_map.items()
        }

        for future in as_completed(futures):
            weather_list.extend(future.result())

    return pd.DataFrame(weather_list)

# =========================================================
# 시각에 따라 과거(관측) / 현재(초단기실황) / 미래(단기예보) 자동 분기
# =========================================================

def fetch_weather_data_auto(target_time, grid_points):
    """
    target_time이 현재 시각보다 충분히 과거(1시간 초과)  → 과거 관측 API
    target_time이 현재 시각과 거의 일치(±1시간 이내)     → 초단기실황 API (최신 관측)
    target_time이 현재 시각보다 미래                       → 단기예보 API
    """

    now = now_kst()
    diff = (target_time - now).total_seconds()

    # 미래
    if diff > 3600:
        if target_time > now + timedelta(hours=67):
            return pd.DataFrame()
        return fetch_forecast_data(target_time, grid_points)

    # 현재 (전후 1시간 이내는 "현재"로 취급 — 가장 신선한 데이터 사용)
    if diff > -3600:
        ncst_df = fetch_ncst_data(grid_points)
        if not ncst_df.empty:
            return ncst_df
        # 초단기실황 실패 시 과거 관측으로 폴백
        return fetch_weather_data(target_time)

    # 명확한 과거
    return fetch_weather_data(target_time)

# =========================================================
# 루트
# =========================================================

@app.get("/")
def root():

    return {
        "status": "success",
        "message": "Black Ice API Server"
    }

# =========================================================
# 지역 API
# =========================================================

@app.get("/regions")
def get_regions():

    return {
        "status": "success",
        "regions": regions
    }

# =========================================================
# 기상 관측소 위치 API
# =========================================================

@app.get("/stations")
def get_stations():
    """기상 관측소 위치 목록 반환"""

    try:
        station_df = meta_df.rename(columns={
            "지점":  "asos_id",
            "지점명": "asos_name"
        }).copy()

        station_df["asos_id"]  = pd.to_numeric(station_df["asos_id"],  errors="coerce")
        station_df["asos_lat"] = pd.to_numeric(station_df["위도"], errors="coerce")
        station_df["asos_lon"] = pd.to_numeric(station_df["경도"], errors="coerce")

        station_df = station_df.dropna(
            subset=["asos_id", "asos_lat", "asos_lon"]
        )

        result = station_df[["asos_id", "asos_name", "asos_lat", "asos_lon"]].copy()
        result["asos_id"] = result["asos_id"].astype(int).astype(str)

        return {
            "status": "success",
            "count": len(result),
            "stations": dataframe_to_json_records(result)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "stations": []
        }

# =========================================================
# 예측 API (지역 선택 기반)
# =========================================================

@app.post("/predict")
def predict(
    req: PredictRequest
):

    try:

        target_time = datetime.strptime(
            f"{req.date} {req.time}",
            "%Y-%m-%d %H:%M"
        )

        selected_df = base_df[

            (base_df["시도"] == req.province)

            &

            (base_df["시군구"] == req.city)

        ].copy()

        if selected_df.empty:

            return {
                "status": "error",
                "message": "지역 데이터 없음",
                "results": []
            }

        selected_df = selected_df.head(
            req.max_points
        )

        selected_df["asos_id"] = \
            selected_df["asos_id"].astype(str)

        # 선택된 지역의 좌표 목록 (단기예보 호출 시 격자 변환용)
        grid_points = list(zip(
            selected_df["asos_id"],
            selected_df["위도"],
            selected_df["경도"]
        ))

        weather_df = fetch_weather_data_auto(
            target_time,
            grid_points
        )

        if weather_df.empty:

            now = now_kst()
            if target_time > now + timedelta(hours=67):
                msg = "선택하신 시각은 예보 가능 범위(최대 약 3일 후)를 벗어났습니다."
            else:
                msg = "기상 데이터 없음"

            return {
                "status": "error",
                "message": msg,
                "results": []
            }

        weather_df["asos_id"] = \
            weather_df["asos_id"].astype(str)

        merged = pd.merge(

            selected_df,

            weather_df,

            on="asos_id",

            how="left"
        )

        numeric_cols = [
            "기온",
            "습도",
            "풍향",
            "풍속",
            "강수량",
            "지면온도"
        ]

        for col in numeric_cols:

            if col in merged.columns:

                merged[col] = pd.to_numeric(
                    merged[col],
                    errors="coerce"
                )

        # 결측 기상값은 가까운 다른 지점의 값으로 우선 보간
        merged = fill_missing_with_nearest(
            merged,
            numeric_cols
        )

        # 보간으로도 못 채운 극히 일부 경우에 한해 최소 안전망 적용
        merged["풍향"] = \
            merged["풍향"].fillna(0)

        merged["풍속"] = \
            merged["풍속"].fillna(1.5)

        merged["강수량"] = \
            merged["강수량"].fillna(0)

        merged["기온"] = \
            merged["기온"].fillna(
                merged["기온"].mean() if merged["기온"].notna().any() else 5.0
            )

        merged["습도"] = \
            merged["습도"].fillna(70)

        merged["지면온도"] = \
            merged["지면온도"].fillna(
                merged["기온"]
            )

        # 노면온도 보정 로직(노트북 add_estimated_road_surface_temp와 동일) 적용
        merged = add_estimated_road_surface_temp(merged)

        merged["aws_거리_km"] = merged["asos_distance_m"] / 1000
        merged["hour"] = target_time.hour

        X_icing = make_model_input(merged, icing_features)

        merged["icing_probability"] = icing_model.predict_proba(X_icing)[:, 1]
        merged["icing_probability_percent"] = merged["icing_probability"] * 100

        merged["결빙확률"] = merged["icing_probability"]

        X_blackice = make_model_input(merged, blackice_features)

        merged["blackice_model_probability"] = blackice_model.predict_proba(X_blackice)[:, 1]

        merged["blackice_probability"] = merged["blackice_model_probability"]
        merged["blackice_probability_percent"] = merged["blackice_model_probability"] * 100

        merged["risk_level"] = merged["blackice_probability"].apply(make_risk_level)

        # 사전 생성된 도로 스내핑 캐시에서 보정 좌표 조회 (요청 시점 OSRM 호출 없음)
        merged = apply_snapped_coords(merged)

        result_cols = [

            "시도",
            "시군구",
            "읍면동",

            "위도",
            "경도",

            "asos_id",
            "asos_name",

            "기온",
            "습도",
            "풍속",
            "강수량",
            "지면온도",
            "추정노면온도",

            "icing_probability",
            "icing_probability_percent",



            "blackice_probability",
            "blackice_probability_percent", 
            "risk_level"

        ]

        result_cols = [

            c for c in result_cols

            if c in merged.columns

        ]

        result_df = merged[
            result_cols
        ].copy()

        result_df = result_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return {

            "status": "success",

            "target_time":
                target_time.strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "province":
                req.province,

            "city":
                req.city,

            "count":
                len(result_df),

            "results":
                dataframe_to_json_records(
                    result_df
                )
        }

    except Exception as e:

        print(
            "PREDICT ERROR:",
            str(e)
        )

        return {
            "status": "error",
            "message": str(e),
            "results": []
        }

# =========================================================
# 전국 예측 API (조회 가능한 전체 지점 — 줌인 데모용)
# =========================================================

@app.get("/predict/nationwide")
def predict_nationwide(
    offset_minutes: int = 0,
    top_n: int = 20
):
    """
    조회 가능한 전체 지점(base_df 전체)에 대한 블랙아이스 예측

    offset_minutes : 0=현재, 30=30분 후, 60=1시간 후
    top_n           : 위험도 상위 N개만 별도로 표시하기 위한 개수
    """

    try:
        target_time = now_kst() + timedelta(minutes=offset_minutes)

        # 격자 변환용 좌표 목록 (asos_id, 위도, 경도)
        # asos_id 기준 중복 제거 — ASOS 관측소 수만큼만 API 호출하도록 최소화
        unique_asos = base_df.drop_duplicates(subset=["asos_id"])
        grid_points = list(zip(
            unique_asos["asos_id"].astype(str),
            unique_asos["위도"],
            unique_asos["경도"]
        ))

        weather_df = fetch_weather_data_auto(target_time, grid_points)

        if weather_df.empty:
            now = now_kst()
            if target_time > now + timedelta(hours=67):
                msg = "선택하신 시각은 예보 가능 범위(최대 약 3일 후)를 벗어났습니다."
            else:
                msg = "기상 데이터 없음"

            return {
                "status": "error",
                "message": msg,
                "results": [],
                "top_risk": []
            }

        # 필터링 없이 base_df 전체 지점 사용
        nationwide_df = base_df.copy()

        if nationwide_df.empty:
            return {
                "status": "error",
                "message": "지점 데이터 없음",
                "results": [],
                "top_risk": []
            }

        nationwide_df["asos_id"] = nationwide_df["asos_id"].astype(str)
        weather_df["asos_id"] = weather_df["asos_id"].astype(str)

        merged = pd.merge(
            nationwide_df,
            weather_df,
            on="asos_id",
            how="left"
        )

        numeric_cols = ["기온", "습도", "풍향", "풍속", "강수량", "지면온도"]
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # 결측 기상값은 가까운 다른 지점의 값으로 우선 보간
        merged = fill_missing_with_nearest(merged, numeric_cols)

        # 보간으로도 못 채운 극히 일부 경우에 한해 최소 안전망 적용
        merged["풍향"] = merged["풍향"].fillna(0)
        merged["풍속"] = merged["풍속"].fillna(1.5)
        merged["강수량"] = merged["강수량"].fillna(0)
        merged["기온"] = merged["기온"].fillna(merged["기온"].mean() if merged["기온"].notna().any() else 5.0)
        merged["습도"] = merged["습도"].fillna(70)
        merged["지면온도"] = merged["지면온도"].fillna(merged["기온"])

        # 노면온도 보정 로직(노트북 add_estimated_road_surface_temp와 동일) 적용
        merged = add_estimated_road_surface_temp(merged)

        merged["aws_거리_km"] = merged["asos_distance_m"] / 1000
        merged["hour"] = target_time.hour

        X_icing = make_model_input(merged, icing_features)
        merged["icing_probability"] = icing_model.predict_proba(X_icing)[:, 1]
        merged["icing_probability_percent"] = merged["icing_probability"] * 100
        merged["결빙확률"] = merged["icing_probability"]

        X_blackice = make_model_input(merged, blackice_features)
        merged["blackice_model_probability"] = blackice_model.predict_proba(X_blackice)[:, 1]

        merged["blackice_probability"] = merged["blackice_model_probability"]
        merged["blackice_probability_percent"] = merged["blackice_model_probability"] * 100
        merged["risk_level"] = merged["blackice_probability"].apply(make_risk_level)

        # 사전 생성된 도로 스내핑 캐시에서 보정 좌표 조회 (요청 시점 OSRM 호출 없음)
        merged = apply_snapped_coords(merged)

        result_cols = [
            "시도", "시군구", "읍면동",
            "위도", "경도",
            "asos_id", "asos_name",
            "기온", "습도", "풍속", "강수량", "지면온도", "추정노면온도",
            "icing_probability", "icing_probability_percent",
            "blackice_probability", "blackice_probability_percent",
            "risk_level"
        ]

        result_cols = [c for c in result_cols if c in merged.columns]
        result_df = merged[result_cols].copy()

        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        # 결측 좌표 제거
        result_df = result_df.dropna(subset=["위도", "경도"])

        all_results = dataframe_to_json_records(result_df)

        # 위험도 기준 정렬 후 상위 N개 추출
        sorted_results = sorted(
            all_results,
            key=lambda r: r.get("blackice_probability_percent", 0) or 0,
            reverse=True
        )

        top_risk = sorted_results[:top_n]

        return {
            "status": "success",
            "target_time": target_time.strftime("%Y-%m-%d %H:%M"),
            "offset_minutes": offset_minutes,
            "count": len(all_results),
            "results": all_results,
            "top_risk": top_risk
        }

    except Exception as e:
        print("NATIONWIDE PREDICT ERROR:", str(e))
        return {
            "status": "error",
            "message": str(e),
            "results": [],
            "top_risk": []
        }

# =========================================================
# 도로 스내핑 캐시 사전 생성 (1회성 CLI 실행 전용)
#
# 사용법:
#   python main.py --build-snap-cache
#
# base_df의 전체 unique 좌표를 OSRM으로 보정하여
# snap_cache.csv에 저장한다. 전국 약 1,700개 좌표 기준
# 병렬 처리로 약 1분 정도 소요된다.
#
# 이 스크립트를 먼저 실행해 캐시 파일을 만들어 두면,
# 서버는 기동 시 캐시를 즉시 로드만 하고 실제 요청 처리 중에는
# OSRM을 전혀 호출하지 않으므로 응답 속도에 영향이 없다.
# =========================================================

if __name__ == "__main__":
    import sys

    if "--build-snap-cache" in sys.argv:
        print("=" * 60)
        print("도로 스내핑 캐시 사전 생성을 시작합니다.")
        print("=" * 60)
        build_snap_cache(base_df)
        print("완료되었습니다. 서버를 재시작하면 캐시가 자동 로드됩니다.")
    else:
        print("이 스크립트는 uvicorn으로 직접 실행하지 않습니다.")
        print("서버 실행: uvicorn main:app --host 0.0.0.0 --port 8000")
        print("캐시 생성: python main.py --build-snap-cache")
