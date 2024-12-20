from flask import Flask, request, jsonify
from flask_cors import  CORS
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
CORS(app)


# Translation function
def ai_translate(text, source_lang, target_lang):
# here is where the model loads, using the name of the model
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

# this is to turn the text that user inputs into a token
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)

# decodes the token after it's been translated
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

@app.route("/translate", methods=["POST"])
def translate():

    data = request.json
    text = data.get('text')
    lang_pair = data.get('lang_pair')

    if not text or not lang_pair:
        return jsonify({"error": "Please provide text or a valid lang_pair"}), 400

    try:
        source_lang, target_lang = lang_pair.split("-")

        translated_text = ai_translate(text, source_lang, target_lang)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)