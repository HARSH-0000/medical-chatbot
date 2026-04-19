# 🏥 Medical Chatbot — AI-Powered Healthcare Assistant

> An intelligent conversational assistant that delivers accurate, context-aware responses to health-related queries using Large Language Models and NLP techniques.

---

## 🚀 Overview

This project builds a conversational healthcare assistant capable of answering medical questions, assisting users with basic health information, and improving accessibility to medical knowledge. It leverages **LLMs**, **NLP**, and **retrieval mechanisms** to generate meaningful, grounded responses from large medical knowledge sources.

> ⚠️ **Disclaimer:** This chatbot is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## ✨ Features

- 🧠 **LLM-powered responses** — natural, human-like conversations
- 📚 **Context-aware answers** — grounded in structured medical knowledge
- 🔍 **Intelligent query understanding** — via NLP preprocessing
- ⚡ **Real-time interaction** — responsive user-friendly interface
- 🛠️ **Scalable architecture** — designed for RAG, vector DB, and fine-tuning extensions

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
NLP Processing
(Tokenization · Intent Detection · Entity Extraction)
    │
    ▼
LLM / Retrieval Layer
(OpenAI GPT · LangChain · Vector Search)
    │
    ▼
Response Generation
(Prompt Engineering · Context Injection)
    │
    ▼
User Interface
(Streamlit / Flask)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| LLM Orchestration | LangChain |
| LLM Provider | OpenAI GPT |
| UI Framework | Streamlit / Flask |
| Vector Database -Pinecone |
| Knowledge Source | Medical Book PDF (`Medical_book.pdf`) |
| Environment | `venv` + `.env` config |

---

## 📁 Project Structure

```
medical-chatbot/
├── Data/
│   └── Medical_book.pdf          # Core medical knowledge source
├── GEN_AI_PROJECT.egg-info/
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── research/
│   └── trials.ipynb              # Experimentation & prototyping
├── src/
│   ├── __init__.py
│   ├── helper.py                 # Utility functions (PDF loading, chunking, embeddings)
│   └── prompt.py                 # Prompt templates
├── static/
│   └── style.css                 # UI styling
├── templates/
│   └── index.html                # Frontend template (Flask)
├── app.py                        # Main application entry point
├── template.py                   # Project scaffolding script
├── setup.py                      # Package setup
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not committed)
├── .gitignore
├── .gitattributes
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/HARSH-0000/medical-chatbot.git
cd medical-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ▶️ Run the Application

**Streamlit:**
```bash
streamlit run app.py
```

**Flask:**
```bash
python app.py
```

Then open your browser at `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask).

---

## 💡 Example Use Cases

- 💊 General health-related queries
- 🩺 Symptom-based guidance *(non-diagnostic)*
- 📖 Medical information lookup from structured sources
- 🎓 Patient education and health literacy support

---

## 🔥 Future Improvements

- [ ] 🔗 Full RAG pipeline (Retrieval-Augmented Generation) with vector DB
- [ ] 📊 Medical dataset fine-tuning (PubMed, clinical notes)
- [ ] 🧾 Source citation for every generated answer
- [ ] 🧠 Memory-enabled multi-turn conversations
- [ ] 🌐 Cloud deployment (AWS / GCP / Azure)
- [ ] 🔐 Auth layer for personalized user sessions

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, make your changes, and submit a pull request.

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
---

<p align="center">Built with ❤️ and LLMs · Not a substitute for professional medical advice</p>
