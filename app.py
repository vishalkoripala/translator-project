from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu
app = Flask(__name__)

# Load models
models = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en"
}

tokenizers = {}
model_objs = {}

for key, model_name in models.items():
    tokenizers[key] = MarianTokenizer.from_pretrained(model_name)
    model_objs[key] = MarianMTModel.from_pretrained(model_name)

translator = Translator()

# Hugging Face translation
def translate(text, direction):
    tokenizer = tokenizers[direction]
    model = model_objs[direction]

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    output = model.generate(**inputs)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Google translate
def google_translate(text, src, tgt):
    return translator.translate(text, src=src, dest=tgt).text

# ✅ Move this ABOVE usage
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

def calculate_bleu(reference, candidate):
    reference = preprocess(reference)
    candidate = preprocess(candidate)

    reference = [reference.split()]
    candidate = candidate.split()

    return round(sentence_bleu(reference, candidate) * 100, 2)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        src = request.form["src"]
        tgt = request.form["tgt"]

        if not text.strip():
            return render_template("index.html", google="Please enter text", score=None)

        # ✅ Auto detect (safe)
        if src == "auto":
            detected = translator.detect(text).lang
            if detected in ["en", "hi", "te"]:
                src = detected
            else:
                src = "en"

        # ✅ Main translation (FIXED)
        google_result = google_translate(text, src, tgt)

        # ✅ Safe languages for back-translation
        if src not in ["en", "hi", "te"]:
            src = "en"
        if tgt not in ["en", "hi", "te"]:
            tgt = "en"

        # ✅ Back translation
        back_text = google_translate(google_result, tgt, src)

        # ✅ Accuracy
        score = similarity(text, back_text)
        score = round(score * 100, 2)

        flag = score < 75
        # BLEU score
        bleu = calculate_bleu(text, back_text)
        return render_template("index.html",
                       google=google_result,
                       score=score,
                       bleu=bleu,
                       flag=flag)

    return render_template("index.html", score=None, bleu=None)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)