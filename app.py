"""
Ev Fiyat Tahmin Modeli - Jüri Sunum Arayüzü
=============================================
 
Bu dosyayı proje kök dizinine (load/, clean/, model/, config/ klasörleriyle
AYNI seviyeye) kopyala ve oradan çalıştır:
 
    pip install -r requirements.txt
    python app.py
 
Gerekli: artifacts/ klasöründe final_model.pkl, preprocessor.pkl,
feature_selector.pkl dosyalarının olması (yani train.py daha önce
çalıştırılmış olmalı). Model: ElasticNet (alpha=0.0005, l1_ratio=0.5).
 
LLM ÖZELLİKLERİ (Groq API):
API key artık ekranda görünmüyor, otomatik olarak proje kök dizinindeki
.env dosyasından okunuyor. Çalıştırmadan önce:
1) pip install -r requirements.txt
2) Proje kök dizininde .env dosyası oluştur, içine tek satır yaz:
   GROQ_API_KEY=gsk_senin_gercek_key_in
"""
 
import os
from dotenv import load_dotenv
load_dotenv()  # proje kök dizinindeki .env dosyasını otomatik okur
 
import json
import numpy as np
import gradio as gr
import pandas as pd
from openai import OpenAI
 
from load.load import load_train, drop_unwanted_columns, enforce_data_types
from model.model import load_artifacts
from clean.clean import drop_multicollinear
 
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
 
# Key artık kodda YOK. Proje kök dizinindeki .env dosyasından okunuyor
# (bkz. dosyanın en üstündeki açıklama). .env dosyası .gitignore'da olduğu
# için GitHub'a asla yüklenmez.
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
 
 
def get_groq_client(api_key):
    if not api_key or not api_key.strip():
        return None
    return OpenAI(api_key=api_key.strip(), base_url=GROQ_BASE_URL)
 
 
# ---------------------------------------------------------------------
# 1) "Tipik ev" baz satırını eğitim verisinden oluştur
# ---------------------------------------------------------------------
print("Baz (tipik ev) profili hazırlanıyor...")
 
raw_train = load_train()
X = drop_unwanted_columns(raw_train)
X = enforce_data_types(X)
if "SalePrice" in X.columns:
    X = X.drop(columns=["SalePrice"])
 
MODEL, preprocessor, FEATURE_SELECTOR = load_artifacts()
# Kaydedilmiş preprocessor'ın fit edilmiş cleaner'ını kullanıyoruz ki
# NaN doldurma mantığı (median/mode/None/0) production ile birebir tutarlı olsun.
X_clean = preprocessor.cleaner.transform(X)
 
base_row = {}
for col in X_clean.columns:
    if X_clean[col].dtype.kind in "if":  # int veya float
        base_row[col] = X_clean[col].median()
    else:
        base_row[col] = X_clean[col].mode()[0]
 
 
def bounds(col, as_int=True):
    lo, hi, med = X_clean[col].min(), X_clean[col].max(), X_clean[col].median()
    if as_int:
        return int(lo), int(hi), int(med)
    return float(lo), float(hi), float(med)
 
 
ovq_min, ovq_max, ovq_def = bounds("OverallQual")
gla_min, gla_max, gla_def = bounds("GrLivArea")
tbs_min, tbs_max, tbs_def = bounds("TotalBsmtSF")
gc_min, gc_max, gc_def = bounds("GarageCars")
yb_min, yb_max, yb_def = bounds("YearBuilt")
yr_min, yr_max, yr_def = bounds("YearRemodAdd")
trg_min, trg_max, trg_def = bounds("TotRmsAbvGrd")
la_min, la_max, la_def = bounds("LotArea")
oc_min, oc_max, oc_def = 1, 10, int(X_clean["OverallCond"].median())
 
quality_options = ["Po", "Fa", "TA", "Gd", "Ex"]
mszoning_options = ["RL", "RM", "C (all)"]
mszoning_default = base_row["MSZoning"] if base_row["MSZoning"] in mszoning_options else "RL"
central_air_options = {"Var": "Y", "Yok": "N"}
central_air_default = "Var" if base_row["CentralAir"] == "Y" else "Yok"
 
# NOT: Neighborhood, Fireplaces (sayı), FullBath ve GarageType bilinçli olarak
# kaldırıldı. RandomForest tabanlı FeatureSelector bu sütunları "seçilebilir"
# işaretlese de, asıl tahmini yapan Lasso modelinin bu sütunlara verdiği
# katsayı ~0 çıktı (L1 regularization onları sıfırlamış) — yani demoda
# değiştirseniz bile fiyat gerçekte kıpırdamıyordu. Bunların yerine, Lasso'nun
# gerçek katsayılarını kontrol ederek gerçekten etkili olan MSZoning (İmar
# Bölgesi), OverallCond (Bakım Durumu) ve CentralAir (Merkezi Klima) kondu.
 
 
# ---------------------------------------------------------------------
# 2) Tahmin + katkı hesaplama (LLM'e vermeden ÖNCE gerçek sayılar burada üretilir)
# ---------------------------------------------------------------------
def transform_row(row_dict):
    df = pd.DataFrame([row_dict])
    df = drop_unwanted_columns(df)
    df = enforce_data_types(df)
    df = preprocessor.transform(df)
    df = drop_multicollinear(df)
    df = FEATURE_SELECTOR.transform(df)
    return df
 
 
def predict_with_contributions(row_dict):
    """
    Fiyatı ve her seçili özelliğin tahmine olan yaklaşık dolar etkisini döndürür.
    Etki, ElasticNet'in GERCEK katsayılarından hesaplanır (LLM'in uydurması değil):
    o özelliğin log-fiyata katkısı çıkarılırsa fiyat ne olurdu, farkı hesaplanır.
    """
    X_row = transform_row(row_dict)
    log_price = MODEL.predict(X_row)[0]
    price = float(np.expm1(log_price))
 
    contributions = X_row.iloc[0].values * MODEL.coef_
    dollar_impact = {}
    for col, c in zip(X_row.columns, contributions):
        if abs(c) < 1e-6:
            continue
        price_without = float(np.expm1(log_price - c))
        dollar_impact[col] = price - price_without
 
    top = sorted(dollar_impact.items(), key=lambda x: -abs(x[1]))[:6]
    return price, top
 
 
# ---------------------------------------------------------------------
# 3) Baz (tipik) evin fiyatı - karşılaştırma referansı
# ---------------------------------------------------------------------
BASE_PRICE, _ = predict_with_contributions(base_row)
print(f"Tipik ev tahmini fiyat: ${BASE_PRICE:,.0f}")
 
 
def build_row(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
              year_built, year_remod, overall_cond, tot_rms, lot_area,
              mszoning, central_air_label, kitchen_qual, exter_qual):
    row = base_row.copy()
    row["OverallQual"] = overall_qual
    row["GrLivArea"] = gr_liv_area
    row["TotalBsmtSF"] = total_bsmt_sf
    row["GarageCars"] = garage_cars
    row["YearBuilt"] = year_built
    row["YearRemodAdd"] = year_remod
    row["OverallCond"] = overall_cond
    row["TotRmsAbvGrd"] = tot_rms
    row["LotArea"] = lot_area
    row["MSZoning"] = mszoning
    row["CentralAir"] = central_air_options[central_air_label]
    row["KitchenQual"] = kitchen_qual
    row["ExterQual"] = exter_qual
    return row
 
 
# ---------------------------------------------------------------------
# 4) Tahmin fonksiyonu
# ---------------------------------------------------------------------
def predict(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
            year_built, year_remod, overall_cond, tot_rms, lot_area,
            mszoning, central_air_label, kitchen_qual, exter_qual):
    row = build_row(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
                     year_built, year_remod, overall_cond, tot_rms, lot_area,
                     mszoning, central_air_label, kitchen_qual, exter_qual)
    price, _ = predict_with_contributions(row)
 
    diff = price - BASE_PRICE
    diff_pct = (diff / BASE_PRICE) * 100
    yon = "daha pahalı" if diff >= 0 else "daha ucuz"
 
    return (
        f"## 💰 Tahmini Satış Fiyatı: ${price:,.0f}\n\n"
        f"Tipik bir eve (${BASE_PRICE:,.0f}) göre "
        f"**${abs(diff):,.0f} ({abs(diff_pct):.1f}%) {yon}**."
    )
 
 
# ---------------------------------------------------------------------
# 5) Chatbot açıklaması: rakamlar Python'dan gelir, LLM sadece anlatır
# ---------------------------------------------------------------------
FEATURE_LABELS = {
    "OverallQual": "Genel Kalite", "GrLivArea": "Yaşam Alanı",
    "TotalBsmtSF": "Bodrum Alanı", "GarageCars": "Garaj Kapasitesi",
    "YearBuilt": "Yapım Yılı", "YearRemodAdd": "Tadilat Yılı",
    "OverallCond": "Genel Bakım Durumu", "TotRmsAbvGrd": "Toplam Oda Sayısı",
    "LotArea": "Arsa Alanı", "KitchenQual": "Mutfak Kalitesi",
    "ExterQual": "Dış Cephe Kalitesi", "MSZoning": "İmar Bölgesi Tipi",
    "CentralAir": "Merkezi Klima",
}
 
 
def explain_prediction(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
                        year_built, year_remod, overall_cond, tot_rms, lot_area,
                        mszoning, central_air_label, kitchen_qual, exter_qual):
    client = get_groq_client(DEFAULT_GROQ_API_KEY)
    if client is None:
        return ("⚠️ Groq API Key bulunamadı. Proje kök dizininde `.env` dosyası olduğundan "
                "ve içinde `GROQ_API_KEY=gsk_...` satırının doğru yazıldığından emin ol, "
                "sonra uygulamayı yeniden başlat.")
 
    row = build_row(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
                     year_built, year_remod, overall_cond, tot_rms, lot_area,
                     mszoning, central_air_label, kitchen_qual, exter_qual)
    price, top_contribs = predict_with_contributions(row)
 
    facts = []
    for col, dollar in top_contribs:
        base_col = col.split("_")[0]
        label = FEATURE_LABELS.get(base_col, base_col)
        yon = "artırıyor" if dollar >= 0 else "azaltıyor"
        facts.append(f"- {label}: fiyatı yaklaşık ${abs(dollar):,.0f} {yon}")
    facts_text = "\n".join(facts) if facts else "- Belirgin bir etki bulunamadı."
 
    prompt = (
        "Sen bir ev fiyat tahmin modelinin sonucunu jüriye açıklayan bir asistansın.\n"
        f"Modelin tahmini satış fiyatı: ${price:,.0f}\n"
        "Modelin (ElasticNet regresyon, alpha=0.0005, l1_ratio=0.5) gerçek katsayılarından hesaplanan, bu evin "
        f"fiyatını en çok etkileyen faktörler (yaklaşık dolar etkisiyle):\n{facts_text}\n\n"
        "Bu bilgileri kullanarak jüriye 3-4 cümlelik, sade ve doğal bir Türkçe "
        "açıklama yaz. SADECE yukarıdaki gerçek sayılara dayan, ekstra bilgi uydurma."
    )
 
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API hatası: {e}"
 
 
# ---------------------------------------------------------------------
# 6) Doğal dilden ev tarifi -> slider'ları otomatik doldurma
# ---------------------------------------------------------------------
def parse_description(text):
    client = get_groq_client(DEFAULT_GROQ_API_KEY)
    if client is None:
        return None, ("⚠️ Groq API Key bulunamadı. Proje kök dizininde `.env` dosyası "
                       "olduğundan ve içinde `GROQ_API_KEY=gsk_...` satırının doğru "
                       "yazıldığından emin ol, sonra uygulamayı yeniden başlat.")
    if not text or not text.strip():
        return None, "Önce bir ev tarifi yazmalısın."
 
    schema_prompt = f"""Sen bir bilgi çıkarma (extraction) aracısın. Görevin, SADECE metinde
açıkça yazan bilgileri JSON'a çevirmek. Metinde yazmayan hiçbir şeyi tahmin etme,
uydurma ya da "tipik" bir değer atama.
 
KURAL: Bir alan hakkında metinde açık bir ifade yoksa, o alanı JSON'a HİÇ EKLEME.
Bu kural her alan için ayrı ayrı geçerli — 13 alanın hepsini doldurman GEREKMİYOR,
çoğu zaman 2-3 alan bile doğru bir çıktıdır.
 
EK KURAL — BAĞLAMSIZ SAYILAR: Metinde geçen bir sayı, hangi özelliğe ait olduğu
AÇIKÇA belli değilse (bir birim, sıfat ya da bağlam kelimesi olmadan tek başına
duruyorsa) kesinlikle hiçbir alana atama. "3 oda" → tot_rms=3 (açık). Ama sadece
"3" ya da "150 200" gibi hangi özelliğe ait olduğu belirtilmemiş, rastgele
sıralanmış sayılar → HİÇBİR ALANA ATANMAZ, bu tür sayıları tamamen yok say.
 
Eğer metin bir evle hiç ilgili değilse, anlamsızsa, ya da hiçbir somut bilgi
içermiyorsa, SADECE boş bir JSON nesnesi döndür: {{}}
 
İzin verilen alanlar (hepsi OPSİYONEL, sadece emin olduklarını yaz):
- overall_qual: 1-10 arası tam sayı (genel kalite, 10=mükemmel)
- overall_cond: 1-10 arası tam sayı (bakım/durum, 10=mükemmel)
- gr_liv_area: sqft cinsinden yaşam alanı (tam sayı)
- total_bsmt_sf: sqft cinsinden bodrum alanı (tam sayı)
- garage_cars: 0-4 arası tam sayı (garaj araç kapasitesi)
- year_built: yapım yılı (tam sayı)
- year_remod: son tadilat yılı (tam sayı)
- tot_rms: toplam oda sayısı (tam sayı)
- lot_area: sqft cinsinden arsa alanı (tam sayı)
- mszoning: "RL", "RM", "C (all)" değerlerinden biri (RL=konut/düşük yoğunluk, RM=konut/orta yoğunluk, C (all)=ticari bölge)
- central_air: "Y" veya "N"
- kitchen_qual: "Po","Fa","TA","Gd","Ex" (Ex=mükemmel)
- exter_qual: "Po","Fa","TA","Gd","Ex" (Ex=mükemmel)
 
ÖRNEK 1
Metin: "3 odalı, 2018 yapımı bir ev"
Doğru çıktı: {{"tot_rms": 3, "year_built": 2018}}
(Diğer 11 alan metinde yok, bu yüzden JSON'a eklenmedi.)
 
ÖRNEK 2
Metin: "asdkjaskjd merhaba nasılsın bugün hava güzel"
Doğru çıktı: {{}}
(Ev ile ilgili hiçbir bilgi yok, bu yüzden tamamen boş.)
 
ÖRNEK 3
Metin: "büyük ve güzel bir ev olsun"
Doğru çıktı: {{}}
("Büyük" ve "güzel" belirsiz/öznel ifadeler, somut bir sayıya veya kategoriye
karşılık gelmiyor — tahmin etme, boş bırak.)
 
ÖRNEK 4
Metin: "123 456 789 asdasd"
Doğru çıktı: {{}}
(Sayılar var ama hangi özelliğe ait oldukları belirtilmemiş — bağlamsız, rastgele
sayıları HİÇBİR alana atama, tamamen yok say.)
 
Şimdi kullanıcının gerçek tarifini analiz et.
Kullanıcının tarifi: "{text}"
 
SADECE geçerli JSON döndür, başka hiçbir metin ekleme."""
 
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": schema_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw), None
    except Exception as e:
        return None, f"⚠️ Groq API/JSON hatası: {e}"
 
 
def _clip_int(value, lo, hi, current):
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return current
 
 
# Her alan için metinde GERÇEKTEN geçmesi gereken anahtar kelimeler.
# LLM bir alanı doldursa bile, bu kelimelerden hiçbiri metinde yoksa alan reddedilir.
# Bu, modelin bağlamsız sayıları/tahminleri bir alana atamasına karşı deterministik
# bir güvenlik katmanıdır — hangi LLM kullanılırsa kullanılsın çalışır.
KEYWORD_GATE = {
    "overall_qual": ["kalite", "kaliteli", "kalitesiz", "lüks", "üst düzey",
                     "mükemmel", "harika", "vasat", "kalitesinde"],
    "overall_cond": ["bakım", "bakımlı", "bakımsız", "durum", "durumda",
                      "yıpranmış", "sağlam", "eski", "yenilenmiş", "harabe",
                      "haldeki", "durumundaki"],
    "gr_liv_area": ["yaşam alan", "metrekare", "m2", "m²", "büyüklük",
                     "geniş", "küçük", "ferah", "kompakt", "net alan",
                     "brüt", "sqft", "square feet"],
    "total_bsmt_sf": ["bodrum", "bodrumlu", "bodrum kat"],
    "garage_cars": ["garaj", "araçlık", "otopark", "araba", "garajlı", "garajı"],
    "year_built": ["yapım", "yapıldı", "yapılmış", "inşa", "yapım yılı",
                    "kaç yılında", "tarihli", "tarihinde", "senesinde",
                    "yılında yapıl", "model"],
    "year_remod": ["tadilat", "yenile", "renov", "restore", "elden geçir",
                    "tadilatlı", "tarihli tadilat"],
    "tot_rms": ["oda", "odalı", "oda sayısı"],
    "lot_area": ["arsa", "arazi", "parsel", "lot", "arsalı", "arsası"],
    "mszoning": ["imar", "bölge", "zoning", "ticari", "yerleşim",
                  "konut bölgesi", "mahalle", "semt"],
    "central_air": ["klima", "havaland", "soğutma", "merkezi sistem",
                     "ac ", "klimalı"],
    "kitchen_qual": ["mutfak", "mutfağı"],
    "exter_qual": ["cephe", "dış cephe", "dış görünüm", "dıştan", "dış yüzey"],
}
 
 
def keyword_gate(data, text):
    """LLM'in döndürdüğü her alanı, metinde ilgili anahtar kelime geçip
    geçmediğini kontrol ederek doğrular. Kelime yoksa alan sessizce elenir."""
    text_lower = (text or "").lower()
    filtered = {}
    for key, value in data.items():
        keywords = KEYWORD_GATE.get(key)
        if keywords and any(kw in text_lower for kw in keywords):
            filtered[key] = value
    return filtered
 
 
def apply_description(text, overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
                       year_built, year_remod, overall_cond, tot_rms, lot_area,
                       mszoning, central_air_label, kitchen_qual, exter_qual):
    data, err = parse_description(text)
    if err:
        no_change = [gr.update()] * 13
        return no_change + [err]
 
    data = keyword_gate(data, text)
 
    if len(data) >= 10:
        no_change = [gr.update()] * 13
        uyari = (
            "⚠️ LLM tüm alanları doldurmaya çalıştı, bu genelde tarifin çok belirsiz "
            "olduğunun ya da modelin fazla tahmin yaptığının işareti. Güvenlik için "
            "hiçbir değişiklik uygulanmadı. Lütfen daha net, somut bir tarif yaz "
            "(örn. '3 oda, 2015 yapımı, iyi durumda')."
        )
        return no_change + [uyari]
 
    new_overall_qual = _clip_int(data.get("overall_qual"), ovq_min, ovq_max, overall_qual) if "overall_qual" in data else overall_qual
    new_gr_liv_area = _clip_int(data.get("gr_liv_area"), gla_min, gla_max, gr_liv_area) if "gr_liv_area" in data else gr_liv_area
    new_total_bsmt_sf = _clip_int(data.get("total_bsmt_sf"), tbs_min, tbs_max, total_bsmt_sf) if "total_bsmt_sf" in data else total_bsmt_sf
    new_garage_cars = _clip_int(data.get("garage_cars"), gc_min, gc_max, garage_cars) if "garage_cars" in data else garage_cars
    new_year_built = _clip_int(data.get("year_built"), yb_min, yb_max, year_built) if "year_built" in data else year_built
    new_year_remod = _clip_int(data.get("year_remod"), yr_min, yr_max, year_remod) if "year_remod" in data else year_remod
    new_overall_cond = _clip_int(data.get("overall_cond"), oc_min, oc_max, overall_cond) if "overall_cond" in data else overall_cond
    new_tot_rms = _clip_int(data.get("tot_rms"), trg_min, trg_max, tot_rms) if "tot_rms" in data else tot_rms
    new_lot_area = _clip_int(data.get("lot_area"), la_min, la_max, lot_area) if "lot_area" in data else lot_area
 
    new_mszoning = data.get("mszoning") if data.get("mszoning") in mszoning_options else mszoning
    central_air_raw = data.get("central_air")
    new_central_air_label = {"Y": "Var", "N": "Yok"}.get(central_air_raw, central_air_label)
    new_kitchen_qual = data.get("kitchen_qual") if data.get("kitchen_qual") in quality_options else kitchen_qual
    new_exter_qual = data.get("exter_qual") if data.get("exter_qual") in quality_options else exter_qual
 
    updates = [
        gr.update(value=new_overall_qual), gr.update(value=new_gr_liv_area),
        gr.update(value=new_total_bsmt_sf), gr.update(value=new_garage_cars),
        gr.update(value=new_year_built), gr.update(value=new_year_remod),
        gr.update(value=new_overall_cond), gr.update(value=new_tot_rms),
        gr.update(value=new_lot_area), gr.update(value=new_mszoning),
        gr.update(value=new_central_air_label), gr.update(value=new_kitchen_qual),
        gr.update(value=new_exter_qual),
    ]
 
    field_map = {
        "overall_qual": ("Genel Kalite", new_overall_qual),
        "overall_cond": ("Genel Bakım Durumu", new_overall_cond),
        "gr_liv_area": ("Yaşam Alanı", new_gr_liv_area),
        "total_bsmt_sf": ("Bodrum Alanı", new_total_bsmt_sf),
        "garage_cars": ("Garaj Kapasitesi", new_garage_cars),
        "year_built": ("Yapım Yılı", new_year_built),
        "year_remod": ("Tadilat Yılı", new_year_remod),
        "tot_rms": ("Toplam Oda Sayısı", new_tot_rms),
        "lot_area": ("Arsa Alanı", new_lot_area),
        "mszoning": ("İmar Bölgesi", new_mszoning),
        "central_air": ("Merkezi Klima", new_central_air_label),
        "kitchen_qual": ("Mutfak Kalitesi", new_kitchen_qual),
        "exter_qual": ("Dış Cephe Kalitesi", new_exter_qual),
    }
    satirlar = [f"- **{field_map[k][0]}** → {field_map[k][1]}" for k in data.keys() if k in field_map]
    status = (
        f"✅ Tarifinizden {len(satirlar)} özellik çıkarıldı:\n\n" + "\n".join(satirlar)
        if satirlar else "⚠️ Tarifinizden hiçbir özellik net şekilde çıkarılamadı, hiçbir şey değişmedi."
    )
    return updates + [status]
 
 
# ---------------------------------------------------------------------
# 7) Gradio arayüzü
# ---------------------------------------------------------------------
with gr.Blocks(title="Ev Fiyat Tahmin Modeli") as demo:
    gr.Markdown("# 🏠 House Prices — Uçtan Uca Regresyon Pipeline")
 
    gr.Markdown(
        f"Kaydırıcıları hareket ettirdikçe tahmin **anlık** güncellenir. "
        f"Başlangıç değerleri, eğitim verisindeki tipik (medyan/mod) evi temsil eder."
    )
 
    gr.Markdown("### ✨ Ya da bir ev tarif edin, kaydırıcılar otomatik dolsun")
    with gr.Row():
        desc_box = gr.Textbox(
            label="Ev Tarifi",
            placeholder="Örn: 4 odalı, 2015 yapımı, iyi bakımlı, merkezi klimalı, orta büyüklükte bir ev",
            scale=4,
        )
        apply_btn = gr.Button("Tarife Göre Doldur", scale=1, variant="primary")
    apply_status = gr.Markdown()
 
    with gr.Row():
        with gr.Column():
            overall_qual = gr.Slider(ovq_min, ovq_max, value=ovq_def, step=1,
                                      label="Genel Kalite (1=Kötü, 10=Mükemmel)")
            gr_liv_area = gr.Slider(gla_min, gla_max, value=gla_def, step=10,
                                     label="Yaşam Alanı (sqft)")
            total_bsmt_sf = gr.Slider(tbs_min, tbs_max, value=tbs_def, step=10,
                                       label="Bodrum Alanı (sqft)")
            garage_cars = gr.Slider(gc_min, gc_max, value=gc_def, step=1,
                                     label="Garaj Kapasitesi (araç sayısı)")
            year_built = gr.Slider(yb_min, yb_max, value=yb_def, step=1,
                                    label="Yapım Yılı")
            year_remod = gr.Slider(yr_min, yr_max, value=yr_def, step=1,
                                    label="Son Tadilat Yılı")
        with gr.Column():
            overall_cond = gr.Slider(oc_min, oc_max, value=oc_def, step=1,
                                      label="Genel Bakım Durumu (1=Kötü, 10=Mükemmel)")
            tot_rms = gr.Slider(trg_min, trg_max, value=trg_def, step=1,
                                 label="Toplam Oda Sayısı")
            lot_area = gr.Slider(la_min, la_max, value=la_def, step=100,
                                  label="Arsa Alanı (sqft)")
            mszoning = gr.Dropdown(mszoning_options, value=mszoning_default,
                                    label="İmar Bölgesi Tipi (Zoning)")
            central_air_label = gr.Dropdown(list(central_air_options.keys()),
                                             value=central_air_default,
                                             label="Merkezi Klima")
            kitchen_qual = gr.Dropdown(quality_options, value=base_row["KitchenQual"],
                                        label="Mutfak Kalitesi")
            exter_qual = gr.Dropdown(quality_options, value=base_row["ExterQual"],
                                      label="Dış Cephe Kalitesi")
 
    output = gr.Markdown()
 
    explain_btn = gr.Button("🤖 Bu Fiyatı LLM ile Açıkla", variant="secondary")
    explanation_output = gr.Markdown()
 
    inputs = [overall_qual, gr_liv_area, total_bsmt_sf, garage_cars,
              year_built, year_remod, overall_cond, tot_rms, lot_area,
              mszoning, central_air_label, kitchen_qual, exter_qual]
 
    for inp in inputs:
        inp.change(fn=predict, inputs=inputs, outputs=output)
 
    explain_btn.click(fn=explain_prediction, inputs=inputs, outputs=explanation_output)
 
    apply_btn.click(fn=apply_description, inputs=[desc_box] + inputs,
                     outputs=inputs + [apply_status]).then(
        fn=predict, inputs=inputs, outputs=output
    )
 
    demo.load(fn=predict, inputs=inputs, outputs=output)
 
if __name__ == "__main__":
    demo.launch(share=True)