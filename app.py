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
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spacy model for better NLP processing (optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logger.info("Spacy model loaded successfully")
except:
    nlp = None
    logger.warning("Spacy model not available. Using basic NLP processing.")

# Global variables for model
model = None
tokenizer = None

def load_model():
    """Load the T5 model and tokenizer"""
    global model, tokenizer
    try:
        # Try different models in order of preference
        model_options = [
            "mrm8488/t5-base-finetuned-question-generation-ap",
            "valhalla/t5-base-qg-hl",
            "iarfmoose/t5-base-question-generator"
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                tokenizer = T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                
                # Use GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                logger.info(f"Model {model_name} loaded successfully on {device}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if model is None:
            raise Exception("Failed to load any model")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Load model at startup
load_model()

def improved_preprocess_text(text):
    """Less aggressive preprocessing that maintains context"""
    # Remove excessive whitespace and clean up
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Keep most punctuation and structure
    # Only remove very specific unwanted characters
    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
    
    return text

def extract_key_phrases(text):
    """Extract important phrases and entities that could be question targets"""
    if nlp:
        try:
            doc = nlp(text)
            key_phrases = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'TIME', 'DATE']:
                    key_phrases.append(ent.text)
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if 2 <= len(chunk.text.split()) <= 4:  # Keep 2-4 word phrases
                    key_phrases.append(chunk.text)
                    
            return list(set(key_phrases))[:8]  # Return top 8 phrases
        except:
            pass
    
    # Fallback: simple noun extraction
    try:
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN') and len(word) > 3]
        return nouns[:8]
    except:
        return []

def create_context_chunks(text, chunk_size=3):
    """Create overlapping chunks of sentences for better context"""
    sentences = sent_tokenize(text)
    if len(sentences) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(sentences), chunk_size-1):  # Overlap by 1 sentence
        chunk = ' '.join(sentences[i:i+chunk_size])
        if len(chunk.strip()) > 50:  # Only add substantial chunks
            chunks.append(chunk)
    
    return chunks[:6]  # Limit to 6 chunks

def highlight_answer_candidates(text):
    """Highlight potential answer phrases in text for better question generation"""
    key_phrases = extract_key_phrases(text)
    highlighted_texts = []
    
    for phrase in key_phrases[:3]:  # Limit to top 3 phrases
        if phrase.lower() in text.lower() and len(phrase.split()) <= 3:
            highlighted_text = text.replace(phrase, f"<hl> {phrase} <hl>", 1)
            highlighted_texts.append(highlighted_text)
    
    return highlighted_texts if highlighted_texts else [text]

def post_process_question(question):
    """Clean up and improve generated questions"""
    if not question:
        return ""
    
    # Remove any leftover tokens or artifacts
    question = re.sub(r'<.*?>', '', question)
    question = re.sub(r'generate question:?', '', question, flags=re.IGNORECASE)
    question = re.sub(r'question:?', '', question, flags=re.IGNORECASE)
    question = question.strip()
    
    if not question:
        return ""
    
    # Ensure question starts with capital letter
    question = question[0].upper() + question[1:] if len(question) > 1 else question.upper()
    
    # Ensure question ends with question mark
    if not question.endswith('?'):
        question += '?'
    
    # Fix common grammatical issues
    question = re.sub(r'\s+', ' ', question)  # Multiple spaces
    question = re.sub(r'\s([?.!,])', r'\1', question)  # Space before punctuation
    question = re.sub(r'([?.!,])([a-zA-Z])', r'\1 \2', question)  # Missing space after punctuation
    
    return question

def is_valid_question(question, original_text):
    """Validate if the generated question is meaningful and relevant"""
    if not question or len(question) < 8:
        return False
    
    # Check if it's actually a question
    question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom', 'can', 'do', 'does', 'did', 'will', 'would', 'is', 'are', 'was', 'were']
    if not any(question.lower().startswith(word) for word in question_words):
        return False
    
    # Avoid very short or very long questions
    word_count = len(question.split())
    if word_count < 3 or word_count > 20:
        return False
    
    # Check if question contains words from original text (relevance check)
    question_words_set = set(word_tokenize(question.lower()))
    text_words_set = set(word_tokenize(original_text.lower()))
    
    # Remove common words for better relevance check
    stop_words = set(stopwords.words('english'))
    question_content = question_words_set - stop_words
    text_content = text_words_set - stop_words
    
    if len(question_content) == 0:
        return False
    
    # At least 30% of content words should be in original text
    overlap = len(question_content & text_content)
    relevance_score = overlap / len(question_content) if len(question_content) > 0 else 0
    
    return relevance_score >= 0.3

def generate_improved_questions(text, max_questions=5):
    """Improved question generation with better context and formatting"""
    global model, tokenizer
    
    if not model or not tokenizer:
        raise Exception("Model not loaded")
    
    # Preprocess text with less aggressive cleaning
    clean_text = improved_preprocess_text(text)
    
    if len(clean_text.strip()) < 20:
        return []
    
    # Create context chunks instead of single sentences
    text_chunks = create_context_chunks(clean_text, chunk_size=4)
    
    questions = []
    question_set = set()  # To avoid duplicates
    device = next(model.parameters()).device
    
    for chunk in text_chunks:
        if len(questions) >= max_questions:
            break
            
        # Try different input formats
        input_formats = [
            f"generate question: {chunk}",
            f"ask question about: {chunk}",
        ]
        
        # Add highlighted versions for important chunks
        if len(chunk.split()) > 15:  # Only for substantial chunks
            highlighted_chunks = highlight_answer_candidates(chunk)
            for highlighted in highlighted_chunks[:1]:
                input_formats.append(f"generate question: {highlighted}")
        
        for input_format in input_formats:
            if len(questions) >= max_questions:
                break
                
            try:
                # Tokenize input
                input_ids = tokenizer.encode(
                    input_format,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(device)
                
                # Generate with improved parameters
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_length=50,
                        min_length=8,
                        num_beams=4,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=False,  # Set to False for more consistent results
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=2
                    )
                
                # Decode and clean the question
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
                question = post_process_question(question)
                
                # Check quality and avoid duplicates
                if (question and 
                    len(question) > 8 and 
                    question not in question_set and
                    is_valid_question(question, clean_text)):
                    questions.append(question)
                    question_set.add(question)
                    break  # Move to next chunk
                    
            except Exception as e:
                logger.warning(f"Error generating question for chunk: {e}")
                continue
    
    return questions

def fallback_question_generation(text):
    """Simple fallback method if improved method fails"""
    global model, tokenizer
    
    sentences = sent_tokenize(text)
    questions = []
    device = next(model.parameters()).device
    
    for sentence in sentences[:4]:  # Process max 4 sentences
        if len(sentence.split()) < 5:  # Skip very short sentences
            continue
            
        try:
            input_text = f"question: {sentence}"
            input_ids = tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=40,
                    num_beams=3,
                    early_stopping=True
                )
            
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = post_process_question(question)
            
            if question and len(question) > 8:
                questions.append(question)
        except Exception as e:
            logger.warning(f"Error in fallback generation: {e}")
            continue
    
    return questions

def generate_questions(text):
    """Main question generation function (drop-in replacement)"""
    try:
        # Input validation
        if not text or len(text.strip()) < 20:
            return ["Please provide more text (at least 20 characters) for question generation."]
        
        # Limit text length
        if len(text) > 3000:
            text = text[:3000]
            logger.info("Text truncated to 3000 characters")
        
        # Try improved method first
        questions = generate_improved_questions(text, max_questions=6)
        
        # Fallback to simpler method if needed
        if len(questions) < 2:
            logger.info("Using fallback question generation")
            fallback_questions = fallback_question_generation(text)
            questions.extend(fallback_questions)
        
        # Remove duplicates while preserving order
        unique_questions = []
        seen = set()
        for q in questions:
            if q.lower() not in seen:
                unique_questions.append(q)
                seen.add(q.lower())
        
        return unique_questions[:5] if unique_questions else ["Unable to generate meaningful questions from this text. Please try with different content."]
        
    except Exception as e:
        logger.error(f"Error in question generation: {e}")
        return ["Sorry, there was an error generating questions. Please try again."]

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
            if page_num > 20:  # Limit to first 20 pages
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
        # Check if text or file is provided
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({"error": "No text or file provided"}), 400

        input_text = ""

        # If text is provided
        if 'text' in request.form:
            input_text = request.form['text'].strip()

        # If file is provided
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Check file size (5MB limit)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > 5 * 1024 * 1024:  # 5MB
                return jsonify({"error": "File too large. Maximum size is 5MB."}), 400

            # Extract text based on file type
            try:
                if file.filename.lower().endswith('.txt'):
                    input_text = extract_text_from_txt(file)
                elif file.filename.lower().endswith('.pdf'):
                    input_text = extract_text_from_pdf(file)
                else:
                    return jsonify({"error": "Unsupported file format. Please use .txt or .pdf files."}), 400
            except Exception as e:
                return jsonify({"error": f"Error reading file: {str(e)}"}), 400

        # Validate input text
        if not input_text or len(input_text.strip()) < 20:
            return jsonify({"error": "Please provide at least 20 characters of meaningful text."}), 400

        # Generate questions
        questions = generate_questions(input_text)
        
        # Filter out any remaining invalid questions
        valid_questions = [q for q in questions if q and len(q.strip()) > 8 and q.endswith('?')]
        
        if not valid_questions:
            return jsonify({"error": "Unable to generate meaningful questions from this text. Please try with different content."}), 400
        
        return jsonify({
            "questions": valid_questions,
            "metadata": {
                "input_length": len(input_text),
                "questions_generated": len(valid_questions),
                "model_used": "T5-based Question Generation",
                "target_questions": min(12, max(8, len(input_text.split()) // 20))
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
