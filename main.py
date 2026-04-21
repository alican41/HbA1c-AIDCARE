from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost as xgb
import os

# FastAPI nesnesini oluşturalım
app = FastAPI(title="AIDCARE HbA1c Prediction Service")

# Dosya yollarını kontrol edelim
MODEL_PATH = 'aidcare_hba1c_model.json' # .pkl yerine .json kullanımı tavsiye edilir
SCALER_PATH = 'aidcare_scaler.pkl'

# Modeli ve Scaler'ı yükleme fonksiyonu
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model veya Scaler dosyaları bulunamadı! Lütfen önce eğitim hücresini çalıştırın.")
    
    # Modeli XGBoost formatında yükle (Versiyon hatalarını önler)
    bst_model = xgb.XGBRegressor()
    bst_model.load_model(MODEL_PATH)
    
    # Scaler'ı joblib ile yükle
    sc = joblib.load(SCALER_PATH)
    return bst_model, sc

# Başlangıçta yükle
try:
    model, scaler = load_assets()
except Exception as e:
    print(f"Başlatma Hatası: {e}")

# Girdi şablonunu belirleyelim
class PatientData(BaseModel):
    yas: float
    cinsiyet: int  # Erkek: 1, Kadın: 0
    aks: float     # Açlık Kan Şekeri
    tks: float     # Tokluk Kan Şekeri

# Klinik durum belirleme fonksiyonu
def get_hba1c_status(val):
    if val < 5.7:
        return "Diyabet Yok (Normal)"
    elif val < 6.5:
        return "Pre-Diyabet"
    else:
        return "Diyabet"

@app.get("/")
def home():
    return {
        "proje": "AIDCARE HbA1c Tahmin Sistemi",
        "durum": "Çalışıyor",
        "endpoint": "/predict üzerinden POST isteği gönderin"
    }

@app.post("/predict")
def predict_hba1c(data: PatientData):
    try:
        # 1. Girdiyi diziye dönüştür (Eğitimdeki sütun sırasıyla: Yaş, Cinsiyet, AKŞ, TKŞ)
        features = np.array([[data.yas, data.cinsiyet, data.aks, data.tks]])
        
        # 2. Ölçeklendirme uygula
        features_scaled = scaler.transform(features)
        
        # 3. Tahmin yap
        prediction = model.predict(features_scaled)[0]
        
        # 4. Klinik durum ve sonuçları hazırla
        status = get_hba1c_status(prediction)
        
        return {
            "tahmin_edilen_hba1c": round(float(prediction), 2),
            "klinik_durum": status,
            "analiz_sonucu": "Başarılı",
            "uyarı": "Bu sonuç bir yapay zeka tahminidir, kesin tanı için doktor onayı gereklidir."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Çalıştırma talimatı:
# uvicorn main:app --reload