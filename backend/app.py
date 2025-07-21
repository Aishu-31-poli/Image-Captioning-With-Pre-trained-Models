from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

# Load BLIP model for image captioning
print("🔃 Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ BLIP model loaded.")

# Translation function using deep-translator
def translate_caption(text, target_lang):
    if target_lang == 'en':
        return text

    try:
        lang_map = {
            'hi': 'hi',
            'te': 'te',
            'ta': 'ta'
        }
        translated_text = GoogleTranslator(source='en', target=lang_map[target_lang]).translate(text)
        print(f"🌐 {target_lang} caption: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return text  # fallback to English

# Caption generation endpoint
@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty image file'}), 400

    try:
        image = Image.open(image_file).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    language = request.form.get('language', 'en')

    try:
        # Generate English caption
        inputs = processor(images=image, return_tensors="pt")
        output_ids = model.generate(**inputs)
        caption_en = processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"📷 English Caption: {caption_en}")

        # Translate to selected language
        final_caption = translate_caption(caption_en, language)

        return jsonify({'caption': final_caption})
    except Exception as e:
        print(f"❌ Caption generation error: {e}")
        return jsonify({'error': 'Caption generation failed'}), 500

if __name__ == "__main__":
    app.run(debug=True)
