# 🏠 House Prices — Uçtan Uca Regresyon Pipeline

Kaggle'ın **"House Prices: Advanced Regression Techniques"** yarışması üzerine kurulu,
modüler bir makine öğrenmesi pipeline'ı. ElasticNet regresyon modeli ile ev fiyatı tahmini yapar;
Gradio tabanlı interaktif bir web arayüzü ve Groq/LLaMA destekli doğal dil açıklaması sunar.

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 2. API Anahtarı Ayarı

Proje kök dizininde `.env` dosyası oluştur:

```
GROQ_API_KEY=gsk_senin_gercek_key_in
```

> Ücretsiz Groq API key: https://console.groq.com

### 3. Modeli Eğit

```bash
python train.py
```

`artifacts/` klasörüne üç dosya kaydedilir:
- `final_model.pkl` — ElasticNet regresyon modeli
- `preprocessor.pkl` — Fit edilmiş ön işleme pipeline'ı
- `feature_selector.pkl` — RandomForest tabanlı özellik seçici

### 4. Demo Arayüzünü Başlat

```bash
python app.py
```

Tarayıcında `http://localhost:7860` adresini aç.

---

## 📁 Proje Yapısı

```
pipeline/
├── app.py                  # Gradio web arayüzü (demo)
├── train.py                # Model eğitim scripti
├── requirements.txt        # Python bağımlılıkları
├── compare_models.py       # 6 modeli karşılaştıran değerlendirme scripti
├── .env                    # API key (git'e yüklenmez)
│
├── config/
│   └── config.py           # Merkezi yapılandırma (yollar, hedef sütun)
│
├── load/
│   └── load.py             # Veri yükleme ve bölme fonksiyonları
│
├── clean/
│   └── clean.py            # Veri temizleme ve dönüşüm pipeline'ı
│
├── model/
│   └── model.py            # Model kaydetme/yükleme ve inference
│
├── data/
│   ├── train.csv           # Eğitim verisi (Kaggle)
│   └── test.csv            # Test verisi (Kaggle)
│
├── artifacts/              # Eğitim sonrası üretilen model dosyaları
│   ├── final_model.pkl
│   ├── preprocessor.pkl
│   └── feature_selector.pkl
│
└── notebooks/
    └── 01_explore.ipynb    # Keşifsel veri analizi (EDA)
```

---

## 🔄 Pipeline Akışı

```
train.csv
  │
  ├─ enforce_data_types()        MSSubClass, MoSold → string
  ├─ remove_outliers()           GrLivArea > 4000 & SalePrice < 300K
  ├─ drop_unwanted_columns()     Id sütununu kaldır
  ├─ split_features_target()     X ve y ayır
  ├─ log1p(y)                    Hedef değişkeni normalleştir
  ├─ train_val_split()           %80 eğitim / %20 doğrulama
  │
  ├─ FullPreprocessor
  │   ├─ DataCleaner             NaN doldurma (None / 0 / medyan / mod)
  │   ├─ encode_ordinal()        17 sıralı kategorik → sayısal
  │   ├─ OneHotEncoderWrapper    28 nominal kategorik → binary sütunlar
  │   └─ ScalerWrapper           StandardScaler (binary olmayanlar)
  │
  ├─ drop_multicollinear()       GarageArea, 1stFlrSF kaldır
  ├─ FeatureSelector             RandomForest importance ≥ 0.001 → ~46 özellik
  ├─ GridSearchCV(ElasticNet)     5-Fold CV ile en iyi alpha + l1_ratio bulunur
  │
  └─ artifacts/                  Model ve dönüşüm nesneleri kaydedilir
```

---

## 🧠 Model Detayları

| Özellik | Değer |
|---------|-------|
| Algoritma | ElasticNet Regresyon (L1 + L2) |
| Hiperparametre arama | GridSearchCV, 5-Fold CV |
| Alpha adayları | `[0.0001, 0.0005, 0.001, 0.005, 0.01]` |
| l1_ratio adayları | `[0.1, 0.3, 0.5, 0.7, 0.9]` |
| Bulunan en iyi alpha | `0.0005` |
| Bulunan en iyi l1_ratio | `0.5` (L1 + L2 tam dengesi) |
| Hedef dönüşüm | `log1p(SalePrice)` → tahmin: `expm1(prediction)` |
| Validation RMSE (tek bölme) | `0.1177` |
| **5-Fold CV RMSE** | **`0.1227 ± 0.0072`** ← daha güvenilir |
| Seçilen özellik sayısı | `46` |

> **Not:** Tek validation seti RMSE'si (`0.1177`) şanslı bir bölmeye denk geldiği için
> biraz iyimserdi. 5-Fold CV, her seferinde farklı bölmeler kullanarak daha
> gerçekçi bir performans tahmini verir: **`0.1227 ± 0.0072`**.
> Düşük standart sapma (±0.0072), modelin veri bölümüne duyarlı olmadığını
> ve güvenilir biçimde genelleme yaptığını gösteriyor.

---

## 🎨 Arayüz Özellikleri

- **Anlık tahmin** — Slider/dropdown değiştiğinde fiyat anında güncellenir  
- **Katkı analizi** — Her özelliğin fiyata dolar cinsinden etkisi (ElasticNet katsayılarından)  
- **Model karşılaştırması** — `compare_models.py` ile 6 algoritma kıyaslanır, grafik otomatik kaydedilir  
- **LLM açıklaması** — "Bu Fiyatı LLM ile Açıkla" butonu ile Türkçe yorum  
- **Doğal dil → slider** — "4 odalı, 2015 yapımı ev" gibi metin yazınca sliderlar otomatik dolar  

---

## 🤖 LLM Entegrasyonu

- **Model:** `llama-3.3-70b-versatile` (Groq üzerinden)  
- **Hallucination önlemi:** LLM'e sadece modelin gerçek katsayılarından hesaplanan dolar etkileri verilir; ek bilgi üretmesi engellenir  
- **Keyword gate:** Kullanıcı metni anahtar kelime filtresiyle deterministik olarak doğrulanır  

---

## 🔒 Güvenlik

- API key **asla** kaynak koda yazılmamalıdır — yalnızca `.env` dosyasında tutulur  
- `.env` ve `artifacts/` `.gitignore`'da tanımlıdır, repoya yüklenmez  
- `data/*.csv` de repoya yüklenmez (Kaggle lisansı)  

---

## 📦 Veri Seti

Kaggle'dan indirilebilir:  
👉 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

İndirilen `train.csv` ve `test.csv` dosyalarını `data/` klasörüne koy.

---

## 🛠️ Geliştirme Notları

- Alpha ve l1_ratio aday listelerini değiştirmek için [`train.py`](train.py) içindeki `param_grid` düzenlenir  
- Yeni özellik eklemek için [`clean/clean.py`](clean/clean.py) içindeki encoding listelerine eklenir  
- Uygulama varsayılan olarak `share=True` ile başlar (genel erişilebilir URL); yerel kullanım için `app.py` sonundaki `demo.launch(share=False)` yapılabilir  
