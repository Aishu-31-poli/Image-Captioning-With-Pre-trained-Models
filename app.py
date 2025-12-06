from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from deep_translator import GoogleTranslator

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load BLIP model
print("🔃 Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ BLIP model loaded.")

# Translation helper
def translate_caption(text, target_lang):
    if target_lang == "en":
        return text
    try:
        lang_map = {"hi": "hi", "te": "te", "ta": "ta"}
        return GoogleTranslator(source="en", target=lang_map[target_lang]).translate(text)
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return text  # fallback

# Serve homepage
@app.route("/")
def home():
    return render_template("index.html")

# Caption generation API
@app.route("/caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty image file"}), 400

    try:
        image = Image.open(image_file).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    language = request.form.get("language", "en")

    try:
        # Generate English caption
        inputs = processor(images=image, return_tensors="pt")
        output_ids = model.generate(**inputs)
        caption_en = processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"📷 English Caption: {caption_en}")

        # Translate if needed
        final_caption = translate_caption(caption_en, language)

        return jsonify({"caption": final_caption})
    except Exception as e:
        print(f"❌ Caption generation error: {e}")
        return jsonify({"error": "Caption generation failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
