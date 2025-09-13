# AI Question Generation Web App

This repository contains a Flask-based web application that generates meaningful and contextually relevant questions from a given text input or uploaded file (PDF or TXT) using **T5-based question generation models**. The application uses advanced NLP techniques for improved question quality and relevance.

---

## 📌 Table of Contents  

- [🚀 Features](#-features)  
- [🛠️ Installation & Usage](#️-installation--usage)   
- [📊 Workflow](#-workflow)  
- [📌 Future Enhancements](#-Future-Enhancements).
- [🤝 Contributing](#-contributing)  
- [👨‍💻 Author](#-author)  
- [📜 License](#-license)  
---

## 🚀 Features

- Generate high-quality questions from raw text or uploaded documents.
- Supports `.txt` and `.pdf` file inputs.
- Uses multiple pre-trained T5 models for robust question generation.
- Intelligent text preprocessing and context chunking for better results.
- Highlights key phrases and entities for more focused questions.
- Provides a health check endpoint to ensure the model and tokenizer are loaded.
- Handles errors gracefully with fallback mechanisms for question generation.
- Lightweight and easy-to-deploy Flask web application with a simple frontend.

---

## 🛠️ Installation & Usage 

- **Backend Framework:** Flask
- **NLP & Question Generation:**
  - Transformers (`T5ForConditionalGeneration`, `T5Tokenizer`) from Hugging Face
  - NLTK for text tokenization and POS tagging
  - SpaCy (optional) for advanced NLP processing (named entities and noun chunks)
- **PDF Processing:** PyPDF2
- **Logging:** Python `logging` module for monitoring and debugging
- **Python Version:** 3.10+
- **Frontend:** HTML templates rendered via Flask (`index.html`, `about.html`, `help.html`)

---


## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/ai-question-generator.git
cd ai-question-generator
```

### 2️⃣ Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK Resources
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### 5️⃣ (Optional) Download SpaCy model for better NLP processing
```bash
python -m spacy download en_core_web_sm
```

### 6️⃣ Run Flask application
```bash
python app.py
```

### 7️⃣ Open in browser
```bash
http://127.0.0.1:5000/
```

### ⚙️ Usage

1. Navigate to the Home Page

2. Enter text manually OR upload a .txt/.pdf file

3. Click Generate Questions

4. View AI-generated questions instantly

---

## 📊 Workflow   

- Text Preprocessing: Cleans and standardizes text while maintaining context.

- Key Phrase Extraction: Uses NLP techniques to identify important words, entities, and phrases.

- Context Chunking: Splits text into overlapping chunks for better question generation.

Question Generation:

Uses T5 models to generate multiple questions for each chunk.

Post-processes questions for grammar, formatting, and relevance.

Fallback Mechanism: If improved generation fails, uses simpler fallback generation.

Output Filtering: Removes duplicates, irrelevant, or invalid questions.

---

## 📌 Future Enhancements  

- Support for additional file types (e.g., DOCX, HTML).
- Integration with a frontend framework like React for better UX.
- Option to select question difficulty or type (e.g., multiple-choice, short answer).
- Batch processing for multiple documents at once.
- Deployment using Docker and CI/CD pipelines for production readiness.
- Caching commonly used models for faster response time.
- Integration with external APIs for live educational content and question generation.

---


### Generated Questions:

1. What is Artificial Intelligence?

2. How is human intelligence simulated in machines?

3. What are machines programmed to do in AI?

### 📈 Results

✅ Generates clear, relevant, and grammatically correct questions

✅ Works on educational text, research papers, and general articles

✅ Supports multiple input formats

---

## 🤝 Contributing  

Contributions are welcome! 🎉  
- Fork the repo  
- Create a new branch (`feature-xyz`)  
- Commit your changes  
- Submit a Pull Request  

---

## 👨‍💻 Author  

**Jaineel Purani**  

📌 [GitHub](https://github.com/jaineel555)  
📌 [LinkedIn](https://www.linkedin.com/in/jaineel-purani-9a128120b/)  
📌 [Instagram](https://www.instagram.com/jaineel_purani__555/)  
📌 [Email](mailto:jaineelpurani555@gmail.com)  

---

## 📜 License  

This project is licensed under the **MIT License** – feel free to use and improve it with giving credits!  

---
