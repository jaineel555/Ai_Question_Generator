from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import PyPDF2
from io import StringIO
import logging
import time

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Route to render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Home page route
@app.route('/home')
def home():
    return render_template('index.html')

# About page route
@app.route('/about')
def About():
    return render_template('about.html')

# Help page route
@app.route('/help')
def help():
    return render_template('help.html')

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Global variables for model
model = None
tokenizer = None

def load_model():
    """Load the T5 model and tokenizer"""
    global model, tokenizer
    try:
        model_name = "valhalla/t5-base-qg-hl"
        logger.info(f"Loading model: {model_name}")
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # Set to evaluation mode for faster inference
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Load model at startup
load_model()

def clean_question(question):
    """Clean up generated questions"""
    if not question:
        return ""
    
    # Remove generation prompts
    question = re.sub(r'generate question:?|question:?|ask:?', '', question, flags=re.IGNORECASE)
    question = question.strip()
    
    if not question:
        return ""
    
    # Capitalize first letter
    question = question[0].upper() + question[1:] if len(question) > 1 else question.upper()
    
    # Add question mark if missing
    if not question.endswith('?'):
        question += '?'
    
    # Clean up spacing
    question = re.sub(r'\s+', ' ', question)
    question = re.sub(r'\s+([?.!,])', r'\1', question)
    
    return question

def is_valid_question(question):
    """Quick validation for questions"""
    if not question or len(question) < 8:
        return False
    
    # Check if it starts with a question word or ends with ?
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'do', 'does', 'did', 'will', 'would', 'is', 'are', 'was', 'were']
    
    starts_correctly = any(question.lower().startswith(word) for word in question_words)
    ends_correctly = question.endswith('?')
    
    if not (starts_correctly or ends_correctly):
        return False
    
    # Check word count (3-25 words)
    word_count = len(question.split())
    return 3 <= word_count <= 25

def generate_questions_simple(text):
    """SIMPLE and FAST question generation - max 10 questions"""
    global model, tokenizer
    
    start_time = time.time()
    device = next(model.parameters()).device
    
    # Clean text and get sentences
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = sent_tokenize(text)
    
    questions = []
    seen_questions = set()
    
    logger.info(f"Generating questions from {len(sentences)} sentences...")
    
    # SIMPLE STRATEGY: One question per sentence (up to 10 sentences)
    for i, sentence in enumerate(sentences[:10]):
        if len(questions) >= 10:  # Hard limit
            break
            
        if len(sentence.split()) < 4:  # Skip very short sentences
            continue
        
        try:
            # Single, simple prompt
            prompt = f"generate question: {sentence}"
            
            # Fast tokenization
            input_ids = tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=200,  # Keep it short
                truncation=True
            ).to(device)
            
            # Fast generation with minimal parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=40,  # Short output
                    min_length=8,
                    num_beams=2,    # Just 2 beams for speed
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and clean
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = clean_question(question)
            
            # Quick validation and duplicate check
            if question and is_valid_question(question):
                question_lower = question.lower()
                if question_lower not in seen_questions:
                    questions.append(question)
                    seen_questions.add(question_lower)
                    logger.info(f"Q{len(questions)}: {question}")
            
        except Exception as e:
            logger.warning(f"Error with sentence {i}: {e}")
            continue
    
    # If we have fewer than 5 questions, try with sentence pairs
    if len(questions) < 5 and len(sentences) > 1:
        for i in range(len(sentences) - 1):
            if len(questions) >= 8:  # Don't exceed 8
                break
                
            sentence_pair = f"{sentences[i]} {sentences[i+1]}"
            
            if len(sentence_pair.split()) > 50:  # Skip very long pairs
                continue
            
            try:
                prompt = f"generate question: {sentence_pair}"
                
                input_ids = tokenizer.encode(
                    prompt,
                    return_tensors="pt",
                    max_length=250,
                    truncation=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_length=40,
                        min_length=8,
                        num_beams=2,
                        early_stopping=True
                    )
                
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
                question = clean_question(question)
                
                if question and is_valid_question(question):
                    question_lower = question.lower()
                    if question_lower not in seen_questions:
                        questions.append(question)
                        seen_questions.add(question_lower)
                        logger.info(f"Pair Q{len(questions)}: {question}")
                
            except Exception as e:
                continue
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    logger.info(f"Generated {len(questions)} questions in {processing_time} seconds")
    
    return questions, processing_time

def generate_questions(text):
    """Main function - simple and fast"""
    try:
        if not text or len(text.strip()) < 20:
            return ["Please provide more text for question generation."], 0
        
        # Limit text length for faster processing
        words = text.split()
        if len(words) > 500:  # Limit to 500 words for speed
            text = ' '.join(words[:500])
        
        # Generate questions
        questions, processing_time = generate_questions_simple(text)
        
        # Ensure we have at least 3 questions
        if len(questions) < 3:
            # Simple fallback
            fallback = [
                "What is the main topic discussed in this text?",
                "What information is provided here?",
                "What can you learn from this passage?"
            ]
            questions.extend(fallback)
            questions = list(dict.fromkeys(questions))  # Remove any duplicates
        
        # Return 7-10 questions max
        final_questions = questions[:10]
        
        return final_questions, processing_time
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return ["Error generating questions. Please try again."], 0

# Function to extract text from a .txt file
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        logger.error(f"Error reading txt file: {e}")
        raise

# Function to extract text from a .pdf file  
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            if page_num > 5:  # Limit to first 5 pages
                break
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        raise

# API endpoint to handle text input and generate questions
@app.route('/generate', methods=['POST'])
def generate():
    try:
        start_time = time.time()
        
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({"error": "No text or file provided"}), 400

        input_text = ""

        # Handle text input
        if 'text' in request.form:
            input_text = request.form['text'].strip()

        # Handle file input
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Check file size
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > 2 * 1024 * 1024:  # 2MB limit for speed
                return jsonify({"error": "File too large. Maximum size is 2MB."}), 400

            # Extract text
            try:
                if file.filename.lower().endswith('.txt'):
                    input_text = extract_text_from_txt(file)
                elif file.filename.lower().endswith('.pdf'):
                    input_text = extract_text_from_pdf(file)
                else:
                    return jsonify({"error": "Unsupported file format. Use .txt or .pdf files."}), 400
            except Exception as e:
                return jsonify({"error": f"Error reading file: {str(e)}"}), 400

        # Validate input
        if not input_text or len(input_text.strip()) < 20:
            return jsonify({"error": "Please provide at least 20 characters of text."}), 400

        # Generate questions
        questions, processing_time = generate_questions(input_text)
        
        # Final validation
        valid_questions = [q for q in questions if q and len(q.strip()) > 5]
        
        if not valid_questions:
            return jsonify({"error": "Unable to generate questions. Please try different text."}), 400
        
        total_time = round(time.time() - start_time, 2)
        
        logger.info(f"Returning {len(valid_questions)} questions in {total_time} seconds")
        
        return jsonify({
            "questions": valid_questions,
            "metadata": {
                "input_length": len(input_text),
                "questions_generated": len(valid_questions),
                "processing_time": f"{processing_time}s",
                "total_time": f"{total_time}s",
                "model": "T5 Question Generation (Fast Mode)"
            }
        })
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({"error": "Internal server error. Please try again."}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
