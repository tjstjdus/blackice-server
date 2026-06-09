# ============================================
# main.py
# ============================================

import os
import joblib
import requests
import numpy as np
import pandas as pd

from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import BallTree

# ============================================
# 기본 경로
# ============================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "결빙_비결빙_전국데이터(최종).csv"
)

META_PATH = os.path.join(
    BASE_DIR,
    "data",
    "META_관측지점정보.csv"
)

ICING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "결빙확률모델.pkl"
)

BLACKICE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "블랙아이스확률모델.pkl"
)

# ============================================
# 기상청 API
# ============================================

KMA_API_KEY = "9jV6iWFlSeC1eolhZdngjw"

PAST_URL = (
    "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
)

# ============================================
# FastAPI
# ============================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 모델 로드 함수
# ============================================

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

            features = list(
                model.get_booster().feature_names
            )

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

        features = list(
            model.get_booster().feature_names
        )

    return model, features

# ============================================
# 모델 불러오기
# ============================================

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

# ============================================
# 데이터 로드
# ============================================

base_df = read_csv_safe(DATA_PATH)
meta_df = read_csv_safe(META_PATH)

def attach_nearest_asos(base_df, meta_df):
    base_df = base_df.copy()
    meta_df = meta_df.copy()

    meta_df = meta_df.rename(columns={
        "지점": "asos_id",
        "지점명": "asos_name",
        "위도": "asos_lat",
        "경도": "asos_lon"
    })

    base_df["위도"] = pd.to_numeric(base_df["위도"], errors="coerce")
    base_df["경도"] = pd.to_numeric(base_df["경도"], errors="coerce")

    meta_df["asos_id"] = pd.to_numeric(meta_df["asos_id"], errors="coerce")
    meta_df["asos_lat"] = pd.to_numeric(meta_df["asos_lat"], errors="coerce")
    meta_df["asos_lon"] = pd.to_numeric(meta_df["asos_lon"], errors="coerce")

    base_df = base_df.dropna(subset=["위도", "경도"]).reset_index(drop=True)
    meta_df = meta_df.dropna(subset=["asos_id", "asos_lat", "asos_lon"]).reset_index(drop=True)

    base_rad = np.radians(base_df[["위도", "경도"]].values)
    meta_rad = np.radians(meta_df[["asos_lat", "asos_lon"]].values)

    tree = BallTree(meta_rad, metric="haversine")
    dist, idx = tree.query(base_rad, k=1)

    earth_radius_km = 6371.0088
    matched = meta_df.iloc[idx[:, 0]].reset_index(drop=True)

    base_df["asos_id"] = matched["asos_id"].astype(int).values
    base_df["asos_name"] = matched["asos_name"].values
    base_df["asos_distance_m"] = dist[:, 0] * earth_radius_km * 1000
    base_df["aws_거리_km"] = dist[:, 0] * earth_radius_km

    return base_df
    
base_df = attach_nearest_asos(base_df, meta_df)
# ============================================
# 숫자형 변환
# ============================================

if "asos_id" in base_df.columns:

    base_df["asos_id"] = pd.to_numeric(
        base_df["asos_id"],
        errors="coerce"
    )

if "지점" in meta_df.columns:

    meta_df["지점"] = pd.to_numeric(
        meta_df["지점"],
        errors="coerce"
    )

# ============================================
# 지역 목록 생성
# ============================================

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

# ============================================
# 입력 스키마
# ============================================

class PredictRequest(BaseModel):

    date: str
    time: str

    province: str
    city: str

    max_points: int = 20

# ============================================
# feature 정리
# ============================================

def make_model_input(
    df,
    feature_cols
):

    X = pd.DataFrame(
        index=df.index
    )

    for col in feature_cols:

        if col in df.columns:

            X[col] = df[col]

        else:

            X[col] = 0

    for col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    ).fillna(0)

    return X

# ============================================
# 위험등급
# ============================================

def make_risk_level(prob):

    if prob >= 0.8:
        return "매우 위험"

    elif prob >= 0.6:
        return "위험"

    elif prob >= 0.3:
        return "주의"

    else:
        return "낮음"

# ============================================
# 기상 API 호출
# ============================================

def fetch_weather_data(
    target_time
):

    target_time = target_time.replace(
        minute=0,
        second=0,
        microsecond=0
    )

    tm = target_time.strftime(
        "%Y%m%d%H%M"
    )

    params = {

        "tm": tm,
        "stn": 0,
        "help": 0,

        "authKey":
            KMA_API_KEY
    }

    response = requests.get(
        PAST_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    rows = []

    for line in response.text.splitlines():

        line = line.strip()

        if (
            not line
            or line.startswith("#")
            or line.startswith("7777")
        ):
            continue

        parts = line.split()

        if len(parts) < 36:
            continue

        rows.append(parts)

    weather = []

    for p in rows:

        weather.append({

            "datetime": p[0],

            "asos_id": p[1],

            "풍향": p[2],
            "풍속": p[3],

            "기온": p[11],
            "습도": p[13],

            "강수량": p[15],

            "지면온도": p[35]
        })

    weather_df = pd.DataFrame(
        weather
    )

    if weather_df.empty:
        return weather_df

    weather_df["datetime"] = \
        pd.to_datetime(

            weather_df["datetime"],

            format="%Y%m%d%H%M",

            errors="coerce"
        )

    numeric_cols = [

        "asos_id",

        "풍향",
        "풍속",

        "기온",
        "습도",

        "강수량",
        "지면온도"
    ]

    for col in numeric_cols:

        weather_df[col] = \
            pd.to_numeric(

                weather_df[col],

                errors="coerce"
            )

    weather_df = weather_df.replace(
        [-99, -99.0, -999, -999.0],
        np.nan
    )

    weather_df["강수량"] = \
        weather_df["강수량"].fillna(0)

    return weather_df

# ============================================
# 지역 목록 API
# ============================================

@app.get("/regions")

def get_regions():

    return {

        "status": "success",

        "regions": regions
    }

# ============================================
# 예측 API
# ============================================

@app.post("/predict")

def predict(
    req: PredictRequest
):

    target_time = datetime.strptime(

        f"{req.date} {req.time}",

        "%Y-%m-%d %H:%M"
    )

    # =====================================
    # 기상 데이터
    # =====================================

    weather_df = fetch_weather_data(
        target_time
    )

    if weather_df.empty:

        return {

            "status": "error",

            "message":
                "기상 데이터 없음"
        }

    # =====================================
    # 지역 필터링
    # =====================================

    selected_df = base_df[

        (base_df["시도"] == req.province)

        &

        (base_df["시군구"] == req.city)

    ].copy()

    print(req.province)
    print(req.city)
    
    print(base_df["시도"].unique()[:20])
    print(base_df["시군구"].unique()[:20])
    
    print(selected_df.shape)

    if selected_df.empty:

        return {

            "status": "error",

            "message":
                "지역 데이터 없음"
        }

    # =====================================
    # 최대 개수
    # =====================================

    selected_df = selected_df.head(
        req.max_points
    )

    # =====================================
    # 기상 merge
    # =====================================

    merged = pd.merge(

        selected_df,

        weather_df,

        on="asos_id",

        how="left"
    )

    # =====================================
    # 추정 노면온도
    # =====================================

    merged["추정노면온도"] = (

        merged["기온"]

        -

        (merged["풍속"] * 0.7)

        -

        (
            (100 - merged["습도"])
            * 0.03
        )
    )

    # =====================================
    # 결빙 확률
    # =====================================

    X_icing = make_model_input(

        merged,

        icing_features
    )

    merged["icing_probability"] = \
        icing_model.predict_proba(
            X_icing
        )[:, 1]

    merged["icing_probability_percent"] = \
        merged["icing_probability"] * 100

    merged["icing_predicted_label"] = \
        (
            merged["icing_probability"]
            >= 0.5
        ).astype(int)

    # =====================================
    # 블랙아이스용 결빙확률 컬럼
    # =====================================

    merged["결빙확률"] = \
        merged["icing_probability"]

    # =====================================
    # 블랙아이스 확률
    # =====================================

    X_blackice = make_model_input(

        merged,

        blackice_features
    )

    merged["blackice_probability"] = \
        blackice_model.predict_proba(
            X_blackice
        )[:, 1]

    merged["blackice_probability_percent"] = \
        merged["blackice_probability"] * 100

    merged["blackice_predicted_label"] = \
        (
            merged["blackice_probability"]
            >= 0.5
        ).astype(int)

    # =====================================
    # 위험등급
    # =====================================

    merged["risk_level"] = \
        merged[
            "blackice_probability"
        ].apply(
            make_risk_level
        )

    # =====================================
    # 결과 컬럼
    # =====================================

    result_cols = [

        "fid",

        "시도",
        "시군구",
        "읍면동",

        "위도",
        "경도",

        "asos_id",
        "asos_name",
        "asos_distance_m",
        "aws_거리_km",

        "datetime",

        "기온",
        "습도",
        "풍향",
        "풍속",

        "강수량",
        "지면온도",
        "추정노면온도",

        "icing_probability",
        "icing_probability_percent",
        "icing_predicted_label",

        "blackice_probability",
        "blackice_probability_percent",
        "blackice_predicted_label",

        "risk_level"
    ]

    result_cols = [

        col for col in result_cols

        if col in merged.columns
    ]

    result_df = merged[
        result_cols
    ].copy()

    result_df = result_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result_df = result_df.where(
        pd.notnull(result_df),
        None
    )

    def clean_json_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (np.float32, np.float64)):
        return float(value)

    if isinstance(value, (np.int32, np.int64)):
        return int(value)

    return value


def dataframe_to_json_records(df):
    records = df.to_dict(orient="records")

    clean_records = []

    for row in records:
        clean_row = {}

        for key, value in row.items():
            clean_row[key] = clean_json_value(value)

        clean_records.append(clean_row)

    return clean_records

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

        "results": dataframe_to_json_records(result_df)
    }

# ============================================
# 기본 API
# ============================================

@app.get("/")

def root():

    return {

        "message":
            "Black Ice Forecast API is running"
    }
