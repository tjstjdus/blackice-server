from fastapi.middleware.cors import CORSMiddleware

import os
from datetime import datetime, timedelta, timezone

import joblib
import requests
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import BallTree


KST = timezone(timedelta(hours=9))

BASE_DIR = "."

TERRAIN_FILE = os.path.join(BASE_DIR, "data", "결빙_비결빙_전국데이터(최종).csv")
ASOS_META_FILE = os.path.join(BASE_DIR, "data", "META_관측지점정보.csv")

ICING_MODEL_FILE = os.path.join(BASE_DIR, "models", "xgb_icing_model.pkl")
BLACKICE_MODEL_FILE = os.path.join(BASE_DIR, "models", "xgb_blackice_model.pkl")

ICING_FEATURE_FILE = os.path.join(BASE_DIR, "models", "icing_feature_cols.pkl")
BLACKICE_FEATURE_FILE = os.path.join(BASE_DIR, "models", "blackice_feature_cols.pkl")

KMA_API_KEY = os.getenv("KMA_API_KEY", "9jV6iWFlSeC1eolhZdngjw")

PAST_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
CURRENT_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_odam_grd"
FUTURE_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"


app = FastAPI(title="Black Ice Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tjstjdus.github.io",
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class PredictRequest(BaseModel):
    date: str
    time: str
    province: str | None = None
    city: str | None = None
    max_points: int | None = 30


def read_csv_safe(path):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise ValueError(f"파일 읽기 실패: {path}")


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "Drainage grade": "Drainage Class",
        "평균기온": "기온",
        "평균습도": "습도",
        "평균풍속": "풍속",
        "평균강수량": "강수량",
        "평균지면온도": "지면온도",
        "평균추정노면온도": "추정노면온도",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def clean_weather_missing_values(df):
    df = df.copy()

    weather_cols = ["기온", "습도", "풍속", "강수량", "지면온도", "추정노면온도"]

    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "기온" in df.columns:
        df["기온"] = df["기온"].replace([-99, -99.0, -999, -999.0], np.nan)

    for col in ["습도", "풍속"]:
        if col in df.columns:
            df[col] = df[col].replace([-9, -9.0, -99, -99.0, -999, -999.0], np.nan)

    if "강수량" in df.columns:
        df["강수량"] = df["강수량"].replace(
            [-9, -9.0, -99, -99.0, -999, -999.0],
            np.nan
        )
        df["강수량"] = df["강수량"].fillna(0)

    if "지면온도" in df.columns:
        df["지면온도"] = df["지면온도"].replace(
            [-9, -9.0, -99, -99.0, -999, -999.0],
            np.nan
        )

    if "풍속" in df.columns:
        df["풍속"] = df["풍속"].fillna(0)

    if "습도" in df.columns:
        hum_median = df["습도"].median(skipna=True)
        if pd.isna(hum_median):
            hum_median = 0
        df["습도"] = df["습도"].fillna(hum_median)

    return df


def attach_nearest_asos(terrain_df, asos_meta_df):
    terrain = normalize_columns(terrain_df)
    asos = normalize_columns(asos_meta_df)

    asos = asos.rename(columns={
        "지점": "asos_id",
        "지점명": "asos_name",
        "위도": "asos_lat",
        "경도": "asos_lon"
    })

    required_terrain = ["위도", "경도"]
    required_asos = ["asos_id", "asos_name", "asos_lat", "asos_lon"]

    missing_terrain = [c for c in required_terrain if c not in terrain.columns]
    missing_asos = [c for c in required_asos if c not in asos.columns]

    if missing_terrain:
        raise KeyError(f"지형데이터에 필요한 컬럼이 없습니다: {missing_terrain}")

    if missing_asos:
        raise KeyError(f"ASOS 메타데이터에 필요한 컬럼이 없습니다: {missing_asos}")

    terrain["위도"] = pd.to_numeric(terrain["위도"], errors="coerce")
    terrain["경도"] = pd.to_numeric(terrain["경도"], errors="coerce")

    asos["asos_lat"] = pd.to_numeric(asos["asos_lat"], errors="coerce")
    asos["asos_lon"] = pd.to_numeric(asos["asos_lon"], errors="coerce")
    asos["asos_id"] = pd.to_numeric(asos["asos_id"], errors="coerce")

    terrain = terrain.dropna(subset=["위도", "경도"]).reset_index(drop=True)
    asos = asos.dropna(subset=["asos_lat", "asos_lon", "asos_id"]).reset_index(drop=True)

    if terrain.empty:
        raise ValueError("위도/경도 결측 제거 후 지형데이터가 비었습니다.")

    if asos.empty:
        raise ValueError("위도/경도 결측 제거 후 ASOS 메타데이터가 비었습니다.")

    terrain_rad = np.radians(terrain[["위도", "경도"]])
    asos_rad = np.radians(asos[["asos_lat", "asos_lon"]])

    tree = BallTree(asos_rad, metric="haversine")
    dist, idx = tree.query(terrain_rad, k=1)

    terrain["asos_id"] = asos.iloc[idx.flatten()]["asos_id"].astype(int).values
    terrain["asos_name"] = asos.iloc[idx.flatten()]["asos_name"].values
    terrain["asos_lat"] = asos.iloc[idx.flatten()]["asos_lat"].values
    terrain["asos_lon"] = asos.iloc[idx.flatten()]["asos_lon"].values
    terrain["asos_distance_m"] = dist.flatten() * 6371000

    return terrain


def dfs_xy_conv(lat, lon):
    RE = 6371.00877
    GRID = 5.0
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136

    DEGRAD = np.pi / 180.0

    re = RE / GRID

    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD

    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = np.log(np.cos(slat1) / np.cos(slat2)) / np.log(
        np.tan(np.pi * 0.25 + slat2 * 0.5)
        /
        np.tan(np.pi * 0.25 + slat1 * 0.5)
    )

    sf = (
        np.tan(np.pi * 0.25 + slat1 * 0.5) ** sn
    ) * np.cos(slat1) / sn

    ro = re * sf / (
        np.tan(np.pi * 0.25 + olat * 0.5) ** sn
    )

    ra = re * sf / (
        np.tan(np.pi * 0.25 + lat * DEGRAD * 0.5) ** sn
    )

    theta = lon * DEGRAD - olon

    if theta > np.pi:
        theta -= 2.0 * np.pi

    if theta < -np.pi:
        theta += 2.0 * np.pi

    theta *= sn

    nx = int(ra * np.sin(theta) + XO + 0.5)
    ny = int(ro - ra * np.cos(theta) + YO + 0.5)

    return nx, ny


def parse_target_time(target_time_str):
    target_time = datetime.strptime(target_time_str, "%Y-%m-%d %H:%M")
    return target_time.replace(tzinfo=KST)


def get_time_type(target_time):
    now = datetime.now(KST)

    if target_time < now - timedelta(hours=2):
        return "past"
    elif target_time <= now + timedelta(minutes=30):
        return "current"
    else:
        return "future"


def fetch_past_asos_all(target_time):
    tm = target_time.strftime("%Y%m%d%H%M")

    params = {
        "tm": tm,
        "stn": 0,
        "help": 0,
        "authKey": KMA_API_KEY
    }

    r = requests.get(PAST_URL, params=params, timeout=30)
    r.raise_for_status()

    rows = []

    for line in r.text.splitlines():
        line = line.strip()

        if not line or line.startswith("#") or line.startswith("7777"):
            continue

        parts = line.split()

        if len(parts) < 36:
            continue

        rows.append(parts)

    data = []

    for p in rows:
        data.append({
            "datetime": p[0],
            "asos_id": p[1],
            "풍향": p[2],
            "풍속": p[3],
            "기온": p[11],
            "습도": p[13],
            "강수량": p[15],
            "지면온도": p[35]
        })

    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M", errors="coerce")
    df["asos_id"] = pd.to_numeric(df["asos_id"], errors="coerce")

    for col in ["풍향", "풍속", "기온", "습도", "강수량", "지면온도"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([-99, -99.0, -999, -999.0], np.nan)
    df["강수량"] = df["강수량"].fillna(0)

    return df


def fetch_current_weather(lat, lon, target_time):
    nx, ny = dfs_xy_conv(lat, lon)
    tm = target_time.strftime("%Y%m%d%H%M")

    params = {
        "tm": tm,
        "nx": nx,
        "ny": ny,
        "authKey": KMA_API_KEY
    }

    r = requests.get(CURRENT_URL, params=params, timeout=30)
    r.raise_for_status()

    result = {
        "기온": np.nan,
        "습도": np.nan,
        "풍속": np.nan,
        "강수량": 0.0,
        "지면온도": np.nan
    }

    for line in r.text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        cat = parts[0]
        val = parts[-1]

        try:
            val = float(val)
        except Exception:
            continue

        if cat in ["T1H", "TA", "TMP"]:
            result["기온"] = val
        elif cat in ["REH", "HM"]:
            result["습도"] = val
        elif cat in ["WSD", "WS"]:
            result["풍속"] = val
        elif cat in ["RN1", "RN", "PCP"]:
            result["강수량"] = val

    return result


def fetch_future_weather(lat, lon, target_time):
    nx, ny = dfs_xy_conv(lat, lon)

    tm = datetime.now(KST).strftime("%Y%m%d%H%M")
    tmef = target_time.strftime("%Y%m%d%H%M")

    params = {
        "tm": tm,
        "tmef": tmef,
        "nx": nx,
        "ny": ny,
        "authKey": KMA_API_KEY
    }

    r = requests.get(FUTURE_URL, params=params, timeout=30)
    r.raise_for_status()

    result = {
        "기온": np.nan,
        "습도": np.nan,
        "풍속": np.nan,
        "강수량": 0.0,
        "지면온도": np.nan
    }

    for line in r.text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        cat = parts[0]
        val = parts[-1]

        try:
            val = float(val)
        except Exception:
            continue

        if cat in ["T1H", "TA", "TMP"]:
            result["기온"] = val
        elif cat in ["REH", "HM"]:
            result["습도"] = val
        elif cat in ["WSD", "WS"]:
            result["풍속"] = val
        elif cat in ["RN1", "RN", "PCP"]:
            result["강수량"] = val

    return result


def add_estimated_road_surface_temp(df):
    df = df.copy()

    for col in ["기온", "풍속", "강수량", "지면온도"]:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["지면온도"] = df["지면온도"].fillna(df["기온"])
    df["풍속"] = df["풍속"].fillna(0)
    df["강수량"] = df["강수량"].fillna(0)

    df["추정노면온도"] = (
        0.7 * df["기온"]
        + 0.2 * df["지면온도"]
        - 0.3 * df["풍속"]
        - 0.1 * df["강수량"]
    )

    return df


def get_model_feature_names(model, fallback_file=None, fallback_list=None):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if fallback_file and os.path.exists(fallback_file):
        return list(joblib.load(fallback_file))

    if fallback_list is not None:
        return list(fallback_list)

    raise ValueError("모델 입력 feature 목록을 찾을 수 없습니다.")


def load_models_and_features():
    icing_model = joblib.load(ICING_MODEL_FILE)
    blackice_model = joblib.load(BLACKICE_MODEL_FILE)

    default_icing_features = [
        "교량", "터널", "ROAD_RANK",
        "alti_mean", "alti_stdev", "alti_min", "alti_max",
        "terrain_mean", "aspect_mean",
        "sun_mean", "sun_stdev", "sun_min",
        "azi_mean", "Drainage Class",
        "기온", "습도", "풍속", "강수량", "지면온도", "추정노면온도",
        "year", "month", "day", "hour"
    ]

    default_blackice_features = [
        "alti_mean", "alti_stdev", "alti_min", "alti_max",
        "terrain_mean", "aspect_mean",
        "sun_mean", "sun_stdev", "sun_min",
        "azi_mean", "Drainage Class",
        "기온", "습도", "풍속", "강수량", "지면온도", "추정노면온도",
        "icing_probability"
    ]

    icing_features = get_model_feature_names(
        icing_model,
        fallback_file=ICING_FEATURE_FILE,
        fallback_list=default_icing_features
    )

    blackice_features = get_model_feature_names(
        blackice_model,
        fallback_file=BLACKICE_FEATURE_FILE,
        fallback_list=default_blackice_features
    )

    return icing_model, blackice_model, icing_features, blackice_features


def make_model_input(df, feature_cols):
    df = normalize_columns(df)

    X = pd.DataFrame(index=df.index)

    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0

    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    return X


def build_realtime_weather_dataset(
    target_time_str,
    province=None,
    city=None,
    max_points=None,
    debug=True
):
    target_time = parse_target_time(target_time_str)
    time_type = get_time_type(target_time)

    terrain = read_csv_safe(TERRAIN_FILE)
    asos_meta = read_csv_safe(ASOS_META_FILE)

    terrain = normalize_columns(terrain)
    asos_meta = normalize_columns(asos_meta)

    if debug:
        print("[1] 원본 지형데이터:", terrain.shape)
        print("[2] ASOS 메타데이터:", asos_meta.shape)

    terrain = attach_nearest_asos(terrain, asos_meta)

    if province is not None and province != "전체":
        terrain = terrain[
            terrain["시도"].astype(str).str.contains(province, na=False)
        ].copy()

    if city is not None and city != "전체":
        terrain = terrain[
            terrain["시군구"].astype(str).str.contains(city, na=False)
        ].copy()

    if terrain.empty:
        raise ValueError("선택한 지역에 해당하는 지점이 없습니다.")

    if max_points is not None:
        terrain = terrain.head(max_points).copy()

    if debug:
        print("[3] ASOS 매칭 및 지역 필터링 후:", terrain.shape)
        print("[4] 조회유형:", time_type)

    weather_list = []

    if time_type == "past":
        past_all = fetch_past_asos_all(target_time)

        if debug:
            print("[5] 과거 ASOS API 결과:", past_all.shape)

        if past_all.empty:
            raise ValueError("과거 ASOS API 조회 결과가 비었습니다.")

        past_all["asos_id"] = pd.to_numeric(
            past_all["asos_id"],
            errors="coerce"
        ).astype("Int64")

        weather_map = past_all.set_index("asos_id")[
            ["기온", "습도", "풍속", "강수량", "지면온도"]
        ].to_dict("index")

        for _, row in terrain.iterrows():
            asos_id = int(row["asos_id"])

            weather = weather_map.get(asos_id, {
                "기온": np.nan,
                "습도": np.nan,
                "풍속": np.nan,
                "강수량": np.nan,
                "지면온도": np.nan
            })

            weather_list.append(weather)

    else:
        grid_cache = {}

        for _, row in terrain.iterrows():
            lat = row["위도"]
            lon = row["경도"]

            nx, ny = dfs_xy_conv(lat, lon)
            grid_key = (nx, ny, time_type)

            try:
                if grid_key in grid_cache:
                    weather = grid_cache[grid_key]
                else:
                    if time_type == "current":
                        weather = fetch_current_weather(lat, lon, target_time)
                    else:
                        weather = fetch_future_weather(lat, lon, target_time)

                    grid_cache[grid_key] = weather

            except Exception:
                weather = {
                    "기온": np.nan,
                    "습도": np.nan,
                    "풍속": np.nan,
                    "강수량": np.nan,
                    "지면온도": np.nan
                }

            weather_list.append(weather)

        if debug:
            print("[5] 현재/미래 API 호출 격자 수:", len(grid_cache))

    weather_df = pd.DataFrame(weather_list)
    weather_df = clean_weather_missing_values(weather_df)

    if "지면온도" in weather_df.columns and "기온" in weather_df.columns:
        weather_df["지면온도"] = weather_df["지면온도"].fillna(weather_df["기온"])

    if weather_df.empty:
        raise ValueError("weather_df가 비었습니다.")

    df = pd.concat(
        [terrain.reset_index(drop=True), weather_df.reset_index(drop=True)],
        axis=1
    )

    df = add_estimated_road_surface_temp(df)

    df["조회시간"] = target_time_str
    df["조회유형"] = time_type

    df["year"] = target_time.year
    df["month"] = target_time.month
    df["day"] = target_time.day
    df["hour"] = target_time.hour

    if debug:
        print("[7] 지형+기상 결합 결과:", df.shape)

    return df


def add_prediction_columns(df, debug=True):
    df = normalize_columns(df)

    icing_model, blackice_model, icing_features, blackice_features = load_models_and_features()

    X_icing = make_model_input(df, icing_features)

    df["icing_probability"] = icing_model.predict_proba(X_icing)[:, 1]
    df["icing_probability_percent"] = df["icing_probability"] * 100
    df["icing_predicted_label"] = (df["icing_probability"] >= 0.5).astype(int)

    X_blackice = make_model_input(df, blackice_features)

    df["blackice_probability"] = blackice_model.predict_proba(X_blackice)[:, 1]
    df["blackice_probability_percent"] = df["blackice_probability"] * 100
    df["blackice_predicted_label"] = (df["blackice_probability"] >= 0.5).astype(int)

    return df


def make_risk_level(prob):
    if pd.isna(prob):
        return "데이터 없음"
    if prob >= 0.8:
        return "매우 위험"
    elif prob >= 0.6:
        return "위험"
    elif prob >= 0.3:
        return "주의"
    else:
        return "낮음"


def make_result(df):
    df = df.copy()

    df["risk_level"] = df["blackice_probability"].apply(make_risk_level)

    result_cols = [
        "fid", "시도", "시군구", "읍면동", "위도", "경도",
        "asos_id", "asos_name", "asos_distance_m",
        "조회시간", "조회유형",
        "기온", "습도", "풍속", "강수량", "지면온도", "추정노면온도",
        "icing_probability", "icing_probability_percent", "icing_predicted_label",
        "blackice_probability", "blackice_probability_percent", "blackice_predicted_label",
        "risk_level"
    ]

    available_cols = [c for c in result_cols if c in df.columns]

    return df[available_cols].copy()


def predict_realtime_blackice(
    target_time_str,
    province=None,
    city=None,
    max_points=None,
    return_json=False,
    debug=True
):
    df = build_realtime_weather_dataset(
        target_time_str=target_time_str,
        province=province,
        city=city,
        max_points=max_points,
        debug=debug
    )

    df = add_prediction_columns(df, debug=debug)

    result_df = make_result(df)

    if return_json:
        return result_df.to_dict(orient="records")

    return result_df


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "블랙아이스 실시간 예측 서버 실행 중"
    }


@app.get("/regions")
def get_regions():
    terrain = read_csv_safe(TERRAIN_FILE)
    terrain = normalize_columns(terrain)

    if "시도" not in terrain.columns or "시군구" not in terrain.columns:
        return {
            "status": "error",
            "message": "시도/시군구 컬럼이 없습니다.",
            "columns": list(terrain.columns)
        }

    regions = {}

    for province, group in terrain.groupby("시도"):
        cities = (
            group["시군구"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        regions[str(province)] = cities

    return {
        "status": "success",
        "regions": regions
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        target_time_str = f"{req.date} {req.time}"

        result_json = predict_realtime_blackice(
            target_time_str=target_time_str,
            province=req.province,
            city=req.city,
            max_points=req.max_points,
            return_json=True,
            debug=False
        )

        return {
            "status": "success",
            "target_time": target_time_str,
            "province": req.province,
            "city": req.city,
            "count": len(result_json),
            "results": result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
