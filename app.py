from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import PyPDF2  # For PDF text extraction
from io import StringIO

# Initialize Flask app
app = Flask(__name__)

# Route to render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Home page route
@app.route('/home')
def home():
    return render_template('index.html')

# About page route
@app.route('/About')
def About():
    return render_template('about.html')

# Help page route
@app.route('/help')
def help():
    return render_template('help.html')

# Load a fine-tuned T5 model for question generation
model_name = "valhalla/t5-base-qg-hl"  # Use a larger fine-tuned model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Download NLTK resources at runtime
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    # Remove special characters and numbers (but keep punctuation for context)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    # Tokenize and remove stopwords (optional, can be skipped)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Question generation function
def generate_questions(text):
    # Split the input text into sentences
    sentences = sent_tokenize(text)
    questions = []

    for sentence in sentences:
        # Preprocess the sentence
        preprocessed_text = preprocess_text(sentence)

        # Prepare input in a better format for T5
        input_text = f"generate questions: {preprocessed_text}"  # Use plural "questions"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate questions with Beam Search for better results
        outputs = model.generate(
            input_ids,
            max_length=50,  # Increase max_length for longer questions
            num_return_sequences=1,  # Generate one question per sentence
            num_beams=5,  # Beam search for better quality
            early_stopping=True  # Stop generation early if suitable
        )

        # Decode the output
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        questions.append(question.capitalize() + "?" if not question.endswith("?") else question.capitalize())

    return questions

# Function to extract text from a .txt file
def extract_text_from_txt(file):
    text = file.read().decode('utf-8')  # Decode bytes to string
    return text

# Function to extract text from a .pdf file
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# API endpoint to handle text input and generate questions
@app.route('/generate', methods=['POST'])
def generate():
    # Check if text or file is provided
    if 'file' not in request.files and 'text' not in request.form:
        return jsonify({"error": "No text or file provided"}), 400

    input_text = ""

    # If text is provided
    if 'text' in request.form:
        input_text = request.form['text']

    # If file is provided
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Extract text based on file type
        if file.filename.endswith('.txt'):
            input_text = extract_text_from_txt(file)
        elif file.filename.endswith('.pdf'):
            input_text = extract_text_from_pdf(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Generate questions
        questions = generate_questions(input_text)
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)