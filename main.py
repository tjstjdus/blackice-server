import os
import glob
import joblib
import requests
import numpy as np
import pandas as pd

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
# 위험도
# =========================================================

def make_risk_level(prob):

    if prob >= 0.7:
        return "매우 높음"

    elif prob >= 0.4:
        return "높음"

    elif prob >= 0.2:
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
    """

    base_date, base_time = get_forecast_base(now_kst())

    fcst_date = target_time.strftime("%Y%m%d")
    fcst_hour = target_time.strftime("%H00")

    # nx, ny 기준으로 중복 제거 (같은 격자는 한 번만 호출)
    grid_map = {}
    for asos_id, lat, lon in grid_points:
        nx, ny = latlon_to_grid(lat, lon)
        grid_map.setdefault((nx, ny), []).append(asos_id)

    weather_list = []

    for (nx, ny), asos_ids in grid_map.items():

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
            response = requests.get(FORECAST_URL, params=params, timeout=15)
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
                elif category == "WSD":
                    point_data["풍속"] = safe_float(value)
                elif category == "PCP":
                    # "강수없음" 같은 문자열 처리
                    point_data["강수량"] = safe_float(value) if value not in (
                        "강수없음", None
                    ) else 0.0

            if point_data:
                # 단기예보는 지면온도를 제공하지 않으므로 기온으로 대체
                point_data.setdefault("지면온도", point_data.get("기온"))
                point_data.setdefault("강수량", 0.0)

                for asos_id in asos_ids:
                    row = {"asos_id": str(asos_id)}
                    row.update(point_data)
                    weather_list.append(row)

        except Exception as e:
            print(f"단기예보 API 오류 (nx={nx}, ny={ny}):", str(e))
            continue

    return pd.DataFrame(weather_list)

# =========================================================
# 초단기실황 데이터 (현재 시각 — 가장 최신 관측값)
# =========================================================

def fetch_ncst_data(grid_points):
    """
    grid_points : [(asos_id, lat, lon), ...]
    "현재"는 시시각각 변하므로 target_time을 받지 않고
    항상 now_kst() 기준 최신 발표분을 사용
    """

    base_date, base_time = get_ncst_base(now_kst())

    # nx, ny 기준으로 중복 제거 (같은 격자는 한 번만 호출)
    grid_map = {}
    for asos_id, lat, lon in grid_points:
        nx, ny = latlon_to_grid(lat, lon)
        grid_map.setdefault((nx, ny), []).append(asos_id)

    weather_list = []

    for (nx, ny), asos_ids in grid_map.items():

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
            response = requests.get(NCST_URL, params=params, timeout=15)
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
                elif category == "WSD":
                    point_data["풍속"] = safe_float(value)
                elif category == "RN1":
                    point_data["강수량"] = safe_float(value)

            if point_data:
                point_data.setdefault("지면온도", point_data.get("기온"))
                point_data.setdefault("강수량", 0.0)

                for asos_id in asos_ids:
                    row = {"asos_id": str(asos_id)}
                    row.update(point_data)
                    weather_list.append(row)

        except Exception as e:
            print(f"초단기실황 API 오류 (nx={nx}, ny={ny}):", str(e))
            continue

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

        merged["풍속"] = \
            merged["풍속"].fillna(1.5)

        merged["강수량"] = \
            merged["강수량"].fillna(0)

        merged["기온"] = \
            merged["기온"].fillna(0)

        merged["습도"] = \
            merged["습도"].fillna(70)

        merged["지면온도"] = \
            merged["지면온도"].fillna(
                merged["기온"]
            )

        merged["추정노면온도"] = (
            0.7 * merged["기온"]
            + 0.2 * merged["지면온도"]
            - 0.3 * merged["풍속"]
            - 0.1 * merged["강수량"]
        )

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

        numeric_cols = ["기온", "습도", "풍속", "강수량", "지면온도"]
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        merged["풍속"] = merged["풍속"].fillna(1.5)
        merged["강수량"] = merged["강수량"].fillna(0)
        merged["기온"] = merged["기온"].fillna(0)
        merged["습도"] = merged["습도"].fillna(70)
        merged["지면온도"] = merged["지면온도"].fillna(merged["기온"])

        merged["추정노면온도"] = (
            0.7 * merged["기온"]
            + 0.2 * merged["지면온도"]
            - 0.3 * merged["풍속"]
            - 0.1 * merged["강수량"]
        )

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
