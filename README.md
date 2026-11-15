# 🖼 Full Analiz AI — Rasm va Video Tahlilchisi

**LLaVA-NeXT** ko'rish-tili modeli asosida rasm va videolarni batafsil tahlil qiluvchi Streamlit ilovasi. Rasmdagi obyektlarni aniqlaydi, tavsiflaydi va savollarga javob beradi.

---

## ✨ Imkoniyatlar

- 🖼 **Rasm tahlili:** obyektlarni aniqlash, batafsil tavsif
- 🎬 **Video tahlili:** videodan kadrlar olib tahlil qilish (OpenCV)
- 💬 **Savol-javob:** rasm bo'yicha savol berish
- 📊 **Ishonch darajasi** va tahlil vaqti ko'rsatkichi
- 🔗 **URL orqali rasm** yuklash imkoniyati

---

## 🛠 Texnologiyalar

| Texnologiya | Maqsad |
|---|---|
| **Python 3.10+** | Asosiy til |
| **Streamlit** | Veb-interfeys |
| **transformers** | LLaVA-NeXT modeli |
| **PyTorch** | Model hisoblash |
| **OpenCV** | Video bilan ishlash |
| **Pillow** | Rasm bilan ishlash |

---

## ⚙️ O'rnatish

```bash
git clone https://github.com/dastonbek8747/Full-Analiz-AI.git
cd Full-Analiz-AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Agar Google API ishlatilsa, `.streamlit/secrets.toml.example` dan nusxa oling:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
GOOGLE_API_KEY = "..."
GOOGLE_CX = "..."
```

> 🔒 `secrets.toml` git'ga qo'shilmaydi (`.gitignore`da mavjud).

---

## ▶️ Ishga tushirish

```bash
streamlit run main.py
```

---

## 📁 Loyiha tuzilishi

```
Full-Analiz-AI/
├── main.py                        # Asosiy Streamlit ilova
├── requirements.txt
├── tests/                         # Regressiya testlari
└── .streamlit/
    └── secrets.toml.example       # Namunaviy maxfiy sozlamalar
```

---

## ⚠️ Talablar

- **GPU** tavsiya etiladi (model katta). CPU'da ham ishlaydi, lekin sekin.
- Birinchi ishga tushirishda model **yuklab olinadi** (bir necha GB).

---

## 📄 Litsenziya

Litsenziya ko'rsatilmagan.
