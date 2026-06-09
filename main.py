import os
import glob
import math
import joblib
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.neighbors import BallTree

# ============================================
# FastAPI
# ============================================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 시간
# ============================================

KST = timezone(timedelta(hours=9))


# ============================================
# 기본 경로
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================
# API KEY
# ============================================

KMA_API_KEY = os.getenv("KMA_API_KEY", "여기에_API_KEY")


# ============================================
# 기상청 API
# ============================================

PAST_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"


# ============================================
# 파일 찾기
# ============================================


def find_file(patterns):

    for pattern in patterns:

        files = glob.glob(os.path.join(BASE_DIR, pattern))

        files += glob.glob(os.path.join(BASE_DIR, "data", pattern))

        files += glob.glob(os.path.join(BASE_DIR, "models", pattern))

        if files:
            return files[0]

    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {patterns}")


DATA_PATH = find_file(["결빙_비결빙_전국데이터*.csv"])

META_PATH = find_file(["META_관측지점정보*.csv"])

ICING_MODEL_PATH = find_file(["결빙확률모델*.pkl"])

BLACKICE_MODEL_PATH = find_file(["블랙아이스확률모델*.pkl"])


# ============================================
# CSV 읽기
# ============================================


def read_csv_safe(path):

    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:

        try:

            return pd.read_csv(path, encoding=enc, low_memory=False)

        except UnicodeDecodeError:

            continue

    raise Exception(f"CSV 인코딩 읽기 실패: {path}")


# ============================================
# 데이터 로드
# ============================================

base_df = read_csv_safe(DATA_PATH)

meta_df = read_csv_safe(META_PATH)


# ============================================
# 최근접 ASOS 연결
# ============================================


def attach_nearest_asos(base_df, meta_df):

    base_df = base_df.copy()

    meta_df = meta_df.copy()

    meta_df = meta_df.rename(
        columns={
            "지점": "asos_id",
            "지점명": "asos_name",
            "위도": "asos_lat",
            "경도": "asos_lon",
        }
    )

    base_df["위도"] = pd.to_numeric(base_df["위도"], errors="coerce")

    base_df["경도"] = pd.to_numeric(base_df["경도"], errors="coerce")

    meta_df["asos_id"] = pd.to_numeric(meta_df["asos_id"], errors="coerce")

    meta_df["asos_lat"] = pd.to_numeric(meta_df["asos_lat"], errors="coerce")

    meta_df["asos_lon"] = pd.to_numeric(meta_df["asos_lon"], errors="coerce")

    base_df = base_df.dropna(subset=["위도", "경도"]).reset_index(drop=True)

    meta_df = meta_df.dropna(subset=["asos_id", "asos_lat", "asos_lon"]).reset_index(
        drop=True
    )

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
            features = list(model.get_booster().feature_names)

        return model, features

    model = obj

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        features = list(model.get_booster().feature_names)

    return model, features


icing_obj = joblib.load(ICING_MODEL_PATH)
blackice_obj = joblib.load(BLACKICE_MODEL_PATH)

icing_model, icing_features = get_model_and_features(icing_obj)
blackice_model, blackice_features = get_model_and_features(blackice_obj)


# ============================================
# 지역 목록
# ============================================

regions = {}

for province in sorted(base_df["시도"].dropna().unique()):

    cities = sorted(base_df[base_df["시도"] == province]["시군구"].dropna().unique())

    regions[province] = cities


# ============================================
# Request
# ============================================


class PredictRequest(BaseModel):

    date: str

    time: str

    province: str

    city: str

    max_points: int = 20


# ============================================
# 위험도
# ============================================


def make_risk_level(prob):

    if prob >= 0.7:
        return "매우 높음"

    elif prob >= 0.4:
        return "높음"

    elif prob >= 0.2:
        return "주의"

    else:
        return "낮음"


# ============================================
# 모델 입력 생성
# ============================================


def make_model_input(df, feature_cols):

    X = pd.DataFrame()

    for col in feature_cols:

        if col in df.columns:

            X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        else:

            X[col] = 0

    return X


# ============================================
# JSON 정리
# ============================================


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


# ============================================
# 기상 데이터
# ============================================
def safe_float(x):
    try:
        value = float(x)

        if value in [-9, -9.0, -99, -99.0]:
            return np.nan

        return value

    except:
        return np.nan

def fetch_weather_data(target_time):

    tm = target_time.strftime("%Y%m%d%H00")

    params = {"tm": tm, "stn": "0", "help": "0", "authKey": KMA_API_KEY}

    try:

        response = requests.get(PAST_URL, params=params, timeout=30)

        response.raise_for_status()

        text = response.text

        lines = text.split("\n")

        rows = []
        weather_list = []
        
        for line in lines:

            if line.startswith("#"):
                continue

            if len(line.strip()) == 0:
                continue

            parts = line.split()
            print(len(parts), parts)

            if len(parts) < 12:
                continue
                
            try:
                stn_id = parts[1]
                
                weather_list.append({

                    "asos_id": str(stn_id),

                    "temp":
                        safe_float(parts[11]),

                    "humidity":
                        safe_float(parts[13]),

                    "wind_speed":
                        safe_float(parts[2]),

                    "rainfall":
                        safe_float(parts[3]),

                    "surface_temp":
                        safe_float(parts[38])

                })

            except:
                continue

        weather_df = pd.DataFrame(rows)

        return weather_df

    except Exception as e:

        print("기상청 API 오류:", str(e))

        return pd.DataFrame()


# ============================================
# 루트
# ============================================


@app.get("/")
def root():

    return {"status": "success", "message": "Black Ice API Server"}


# ============================================
# 지역 API
# ============================================


@app.get("/regions")
def get_regions():

    return {"status": "success", "regions": regions}


# ============================================
# 예측 API
# ============================================


@app.post("/predict")
def predict(req: PredictRequest):

    try:
        target_time = datetime.strptime(f"{req.date} {req.time}", "%Y-%m-%d %H:%M")

        weather_df = fetch_weather_data(target_time)

        if weather_df.empty:
            return {"status": "error", "message": "기상 데이터 없음", "results": []}

        selected_df = base_df[
            (base_df["시도"] == req.province) & (base_df["시군구"] == req.city)
        ].copy()

        if selected_df.empty:
            return {"status": "error", "message": "지역 데이터 없음", "results": []}

        selected_df = selected_df.head(req.max_points)

        selected_df["asos_id"] = selected_df["asos_id"].astype(str)
        weather_df["asos_id"] = weather_df["asos_id"].astype(str)

        merged = pd.merge(selected_df, weather_df, on="asos_id", how="left")

        numeric_cols = ["기온", "습도", "풍향", "풍속", "강수량", "지면온도"]

        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        merged["풍속"] = merged["풍속"].fillna(0)
        merged["강수량"] = merged["강수량"].fillna(0)
        merged["기온"] = merged["기온"].fillna(0)
        merged["지면온도"] = merged["지면온도"].fillna(
        merged["기온"]
)

        merged["추정노면온도"] = (
            0.7 * merged["기온"]
            + 0.2 * merged["지면온도"]
            - 0.3 * merged["풍속"]
            - 0.1 * merged["강수량"]
        )

        X_icing = make_model_input(merged, icing_features)

        merged["icing_probability"] = icing_model.predict_proba(X_icing)[:, 1]
        merged["icing_probability_percent"] = merged["icing_probability"] * 100
        merged["icing_predicted_label"] = (merged["icing_probability"] >= 0.5).astype(
            int
        )

        X_blackice = make_model_input(merged, blackice_features)

        merged["blackice_model_probability"] = blackice_model.predict_proba(X_blackice)[
            :, 1
        ]

        merged["blackice_model_probability_percent"] = (
            merged["blackice_model_probability"] * 100
        )

        merged["blackice_probability"] = (
            merged["icing_probability"] * merged["blackice_model_probability"]
        )

        merged["blackice_probability_percent"] = merged["blackice_probability"] * 100

        merged["blackice_predicted_label"] = (
            merged["blackice_probability"] >= 0.5
        ).astype(int)

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
            "풍향",
            "풍속",
            "강수량",
            "지면온도",
            "추정노면온도",
            "icing_probability",
            "icing_probability_percent",
            "blackice_model_probability",
            "blackice_model_probability_percent",
            "blackice_probability",
            "blackice_probability_percent",
            "risk_level",
        ]

        result_cols = [c for c in result_cols if c in merged.columns]

        result_df = merged[result_cols].copy()
        result_df = result_df.replace([np.inf, -np.inf], np.nan)

        return {
            "status": "success",
            "target_time": target_time.strftime("%Y-%m-%d %H:%M"),
            "province": req.province,
            "city": req.city,
            "count": len(result_df),
            "results": dataframe_to_json_records(result_df),
        }

    except Exception as e:
        print("PREDICT ERROR:", str(e))

        return {"status": "error", "message": str(e), "results": []}