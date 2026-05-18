"""
HbA1c Tahmin Modeli — FastAPI Server (Versiyon Bağımsız)
Model: v4 Weighted Blend (LGB 43.8% + CAT 56.2%)
R²=0.7281 | MAE=0.3511 | RMSE=0.5424 | MAPE=5.86%
Port: 4000

Modeller native format ile yüklenir (pickle yok → versiyon sorunu yok):
  - lgb_model.txt   → LightGBM native
  - cat_model.cbm   → CatBoost native
  - model_meta.npy  → scaler + metadata
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model_output"

# ─────────────────────────────────────────
# Model yükleme — native formatlar (versiyon bağımsız)
# ─────────────────────────────────────────
try:
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    m_lgb = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model.txt"))

    m_cat = CatBoostRegressor()
    m_cat.load_model(str(MODEL_DIR / "cat_model.cbm"))

    meta     = np.load(MODEL_DIR / "model_meta.npy", allow_pickle=True).item()
    FEATURES = meta["features"]
    BLEND_W  = meta["blend_w"]
    SC_MEAN  = meta["scaler_mean"]
    SC_STD   = meta["scaler_std"]

    W_LGB = float(BLEND_W[1])  # 0.438
    W_CAT = float(BLEND_W[2])  # 0.562

    print(f"✅ Model yüklendi")
    print(f"   Features ({len(FEATURES)}): {FEATURES}")
    print(f"   Blend: LGB={W_LGB:.3f} | CAT={W_CAT:.3f}")

except Exception as e:
    raise RuntimeError(
        f"Model yüklenemedi: {e}\n"
        f"MODEL_DIR: {MODEL_DIR}\n"
        f"Gerekli dosyalar: lgb_model.txt, cat_model.cbm, model_meta.npy"
    )

# ─────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="HbA1c Tahmin API",
    description=(
        "Açlık/tokluk kan şekeri, yaş ve BMI kullanarak HbA1c (%) tahmin eder.\n\n"
        "**Model**: LGB + CatBoost Weighted Blend  |  **R²** = 0.7281  |  **MAPE** = 5.86%"
    ),
    version="4.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TKS_RATIO  = 1.35
BMI_MEDIAN = 27.2

# ─────────────────────────────────────────
# Şemalar
# ─────────────────────────────────────────
class PredictRequest(BaseModel):
    yas: float = Field(..., ge=18, le=100, description="Yaş (yıl)")
    aks: float = Field(..., ge=50,  le=600, description="Açlık kan şekeri (mg/dL)")
    tks: float = Field(None, ge=50, le=700, description="Tokluk kan şekeri 2.saat (mg/dL) — opsiyonel")
    bmi: float = Field(None, ge=10, le=80,  description="BMI kg/m² — opsiyonel")

    model_config = {
        "json_schema_extra": {
            "example": {"yas": 55, "aks": 145, "tks": 210, "bmi": 28.5}
        }
    }

class PredictResponse(BaseModel):
    hba1c_tahmin : float
    hba1c_yuzde  : str
    risk_grubu   : str
    yorum        : str
    tks_imputed  : bool
    bmi_imputed  : bool
    model_bilgi  : dict

# ─────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────
def scale(vec: np.ndarray) -> np.ndarray:
    return (vec - SC_MEAN) / SC_STD

def build_features(yas, aks, tks, bmi):
    tks_imp = tks is None
    bmi_imp = bmi is None
    tks = tks if tks is not None else aks * TKS_RATIO
    bmi = bmi if bmi is not None else BMI_MEDIAN

    feat = {
        "yas"          : yas,
        "aks"          : aks,
        "tks"          : tks,
        "aks_tks_oran" : aks / (tks + 1e-6),
        "aks_tks_fark" : aks - tks,
        "aks_kare"     : aks ** 2,
        "tks_kare"     : tks ** 2,
        "yas_kare"     : yas ** 2,
        "yuksek_aks"   : float(aks > 126),
        "bmi_yas"      : bmi * yas,
        "tks_imputed"  : float(tks_imp),
    }
    vec = np.array([feat[f] for f in FEATURES], dtype=np.float64).reshape(1, -1)
    return scale(vec), tks_imp, bmi_imp

def interpret(v: float):
    if v < 5.7:
        return "Normal",              "✅ Normal aralıkta (< 5.7%)"
    elif v < 6.5:
        return "Prediyabet",          "⚠️ Prediyabet aralığında (5.7–6.4%)"
    elif v < 8.0:
        return "Diyabet (kontrollü)", "🔴 Diyabet — kontrollü aralık (6.5–7.9%)"
    else:
        return "Diyabet (yüksek)",    "🚨 Diyabet — yüksek risk (≥ 8.0%)"

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/", tags=["Genel"])
def root():
    return {
        "mesaj"   : "HbA1c Tahmin API çalışıyor",
        "versiyon": "v4.0.1",
        "docs"    : "/docs",
        "model"   : "LGB+CatBoost Blend  R²=0.7281  MAPE=5.86%",
    }

@app.get("/health", tags=["Genel"])
def health():
    return {"durum": "ok", "model_yuklendi": True}

@app.post("/predict", response_model=PredictResponse, tags=["Tahmin"])
def predict(req: PredictRequest):
    try:
        X_sc, tks_imp, bmi_imp = build_features(req.yas, req.aks, req.tks, req.bmi)
        p_lgb = float(m_lgb.predict(X_sc)[0])
        p_cat = float(m_cat.predict(X_sc)[0])
        hba1c = round(float(np.clip(W_LGB * p_lgb + W_CAT * p_cat, 3.5, 18.0)), 2)
        risk, yorum = interpret(hba1c)

        return PredictResponse(
            hba1c_tahmin = hba1c,
            hba1c_yuzde  = f"%{hba1c:.2f}",
            risk_grubu   = risk,
            yorum        = yorum,
            tks_imputed  = tks_imp,
            bmi_imputed  = bmi_imp,
            model_bilgi  = {
                "model" : "LGB+CatBoost Weighted Blend",
                "r2"    : 0.7281,
                "mae"   : 0.3511,
                "rmse"  : 0.5424,
                "mape"  : "5.86%",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Tahmin"])
def predict_batch(requests: list[PredictRequest]):
    """Toplu tahmin — max 100 kayıt"""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 kayıt")
    return [predict(r) for r in requests]

@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "versiyon"  : "v4.0.1",
        "metrikler" : {"R2": 0.7281, "MAE": 0.3511, "RMSE": 0.5424, "MAPE": "5.86%"},
        "blend"     : {"LGB": round(W_LGB, 3), "CatBoost": round(W_CAT, 3)},
        "features"  : FEATURES,
        "egitim"    : "NHANES + LAB  (n≈10,455)",
    }

# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=4000, reload=False)
