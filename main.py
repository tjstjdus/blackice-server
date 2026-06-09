# ==============================
# main.py
# ==============================

import os
import joblib
import requests
import numpy as np
import pandas as pd

from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================
# 기본 설정
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

REGION_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "지역데이터.csv"
)

KMA_API_KEY = "여기에_API_KEY"

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
# 모델 로드
# ============================================

def get_model_and_features(obj):

    if isinstance(obj, dict):

        model = obj["model"]

        if "features" in obj:

            features = list(obj["features"])

        elif hasattr(model, "feature_names_in_"):

            features = list(model.feature_names_in_)

        else:

            features = list(
                model.get_booster().feature_names
            )

        return model, features

    model = obj

    if hasattr(model, "feature_names_in_"):

        features = list(model.feature_names_in_)

    else:

        features = list(
            model.get_booster().feature_names
        )

    return model, features


icing_obj = joblib.load(ICING_MODEL_PATH)
blackice_obj = joblib.load(BLACKICE_MODEL_PATH)

icing_model, icing_features = \
    get_model_and_features(icing_obj)

blackice_model, blackice_features = \
    get_model_and_features(blackice_obj)

# ============================================
# 지역 데이터
# ============================================

region_df = pd.read_csv(REGION_DATA_PATH)

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
# 유틸 함수
# ============================================

def make_model_input(df, feature_cols):

    X = pd.DataFrame(index=df.index)

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
        X.median(numeric_only=True)
    ).fillna(0)

    return X


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
# 기상 API
# ============================================

def fetch_past_asos_all(target_time):

    target_time = target_time.replace(
        minute=0,
        second=0,
        microsecond=0
    )

    tm = target_time.strftime("%Y%m%d%H%M")

    params = {
        "tm": tm,
        "stn": 0,
        "help": 0,
        "authKey": KMA_API_KEY
    }

    r = requests.get(
        PAST_URL,
        params=params,
        timeout=30
    )

    r.raise_for_status()

    rows = []

    for line in r.text.splitlines():

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

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    df["asos_id"] = pd.to_numeric(
        df["asos_id"],
        errors="coerce"
    )

    for col in [
        "풍향",
        "풍속",
        "기온",
        "습도",
        "강수량",
        "지면온도"
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [-99, -99.0, -999, -999.0],
        np.nan
    )

    df["강수량"] = df["강수량"].fillna(0)

    return df

# ============================================
# 지역 API
# ============================================

@app.get("/regions")

def get_regions():

    result = {}

    grouped = region_df.groupby("시도")["시군구"]

    for province, cities in grouped:

        result[province] = \
            sorted(cities.unique().tolist())

    return {
        "status": "success",
        "regions": result
    }

# ============================================
# 예측 API
# ============================================

@app.post("/predict")

def predict(req: PredictRequest):

    target_time = datetime.strptime(
        f"{req.date} {req.time}",
        "%Y-%m-%d %H:%M"
    )

    weather_df = fetch_past_asos_all(
        target_time
    )

    if weather_df.empty:

        return {
            "status": "error",
            "message": "기상 데이터 없음"
        }

    region_filtered = region_df[
        (region_df["시도"] == req.province)
        &
        (region_df["시군구"] == req.city)
    ].copy()

    if region_filtered.empty:

        return {
            "status": "error",
            "message": "지역 데이터 없음"
        }

    region_filtered = region_filtered.head(
        req.max_points
    )

    merged = pd.merge(
        region_filtered,
        weather_df,
        on="asos_id",
        how="left"
    )

    merged["추정노면온도"] = (
        merged["기온"]
        -
        (merged["풍속"] * 0.7)
        -
        ((100 - merged["습도"]) * 0.03)
    )

    # =================================
    # 결빙 예측
    # =================================

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

    # =================================
    # 블랙아이스 예측
    # =================================

    merged["결빙확률"] = \
        merged["icing_probability"]

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

    merged["risk_level"] = \
        merged["blackice_probability"].apply(
            make_risk_level
        )

    # =================================
    # 결과 정리
    # =================================

    result_cols = [

        "fid",

        "시도",
        "시군구",
        "읍면동",

        "위도",
        "경도",

        "asos_id",

        "datetime",

        "기온",
        "습도",
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
        c for c in result_cols
        if c in merged.columns
    ]

    results = merged[result_cols].copy()

    results = results.replace(
        [np.inf, -np.inf],
        np.nan
    )

    results = results.where(
        pd.notnull(results),
        None
    )

    return {

        "status": "success",

        "target_time":
            target_time.strftime(
                "%Y-%m-%d %H:%M"
            ),

        "province": req.province,
        "city": req.city,

        "count": len(results),

        "results":
            results.to_dict(
                orient="records"
            )
    }

# ============================================
# 서버 실행
# ============================================

@app.get("/")

def root():

    return {
        "message":
            "Black Ice Forecast API"
    }
