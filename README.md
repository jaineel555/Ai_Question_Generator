# 🤖 AI Question Generation System

## 📌 Overview
The **AI Question Generation System** is a Flask-based web application that leverages **Natural Language Processing (NLP)** and **Transformer models (T5)** to automatically generate meaningful questions from given text or uploaded documents.  
It is designed to assist **students, teachers, and content creators** in generating practice questions and assessments.

---

## 🚀 Features
- Accepts **manual text input** or **file uploads** (`.txt`, `.pdf`)
- Uses **HuggingFace T5 Transformer** for high-quality question generation
- Preprocesses text using **NLTK** (cleaning, stopwords removal, tokenization)
- User-friendly web interface with multiple routes (`Home`, `About`, `Help`)
- Supports **JSON API responses** for integration into other apps
- Provides multiple question variations using **Beam Search**

---

## 📂 Dataset
This system does not require a predefined dataset.  
Instead, it generates questions dynamically from:
- User-input text
- Uploaded documents (`txt` / `pdf`)

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

### 4️⃣ Run Flask application
```bash
python app.py
```

### 5️⃣ Open in browser
```bash
http://127.0.0.1:5000/
```

### ⚙️ Usage

1. Navigate to the Home Page

2. Enter text manually OR upload a .txt/.pdf file

3. Click Generate Questions

4. View AI-generated questions instantly

### 📊 Example Output

### Input Text:

Artificial Intelligence is the simulation of human intelligence in machines that are programmed to think and act like humans.


### Generated Questions:

1. What is Artificial Intelligence?

2. How is human intelligence simulated in machines?

3. What are machines programmed to do in AI?

### 📈 Results

✅ Generates clear, relevant, and grammatically correct questions

✅ Works on educational text, research papers, and general articles

✅ Supports multiple input formats
