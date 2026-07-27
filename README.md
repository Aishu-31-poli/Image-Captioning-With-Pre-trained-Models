# 🖼️ Image Captioning With Pre-trained Models

A Flask-based web application that generates natural-language captions for images using Salesforce's pre-trained **BLIP** (Bootstrapping Language-Image Pre-training) model. Supports multilingual caption translation (Hindi, Telugu, Tamil).

## 🚀 Features
- Upload an image and get an AI-generated caption instantly
- Powered by Hugging Face Transformers (BLIP model)
- Multilingual caption translation via Deep Translator
- Simple web UI built with Flask, HTML, and CSS
- REST API endpoint for programmatic access

## 🛠️ Tech Stack
- Python
- Flask + Flask-CORS
- Hugging Face Transformers (`Salesforce/blip-image-captioning-base`)
- PIL (Pillow)
- Deep Translator (Google Translate)

## ⚙️ Installation

1. Clone the repository
```bash
git clone https://github.com/Aishu-31-poli/Image-Captioning-With-Pre-trained-Models.git
cd Image-Captioning-With-Pre-trained-Models
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
python app.py
```

4. Open your browser at `http://127.0.0.1:5000`

## 📡 API Usage

**Endpoint:** `POST /caption`

**Form Data:**
- `image` (file) — the image to caption
- `language` (string, optional) — `en`, `hi`, `te`, or `ta` (default: `en`)

**Example Response:**
```json
{
  "caption": "a dog running on the beach"
}
```

## 📌 Future Improvements
- Support for more languages
- Deploy on cloud (Render/Heroku/AWS)
- Add caption history/logging



