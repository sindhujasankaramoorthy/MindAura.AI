# 🧠 MindAura — AI-Powered Mental Wellness Journal

MindAura is an AI-powered mental wellness platform that analyzes users' journal entries to provide psychologically informed emotional insights. The system combines multilingual text preprocessing, emotion detection, psychological feature engineering, and Large Language Model (LLM) reasoning to generate supportive, context-aware feedback.

---

## Table of Contents

- [Features](#-features)
- [Architecture](#️-project-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#️-tech-stack)
- [AI Pipeline](#-ai-pipeline)
- [Installation](#️-installation)
- [Running the App](#️-running)
- [Current Progress](#-current-progress)
- [Contributors](#-contributors)

---

## 🚀 Features

| Feature | Description |
|---|---|
| ✍️ Journal-based assessment | Analyzes free-form journal entries for emotional wellness |
| 🌐 Tanglish → English normalization | Corrects and normalizes Tamil-English code-mixed text |
| 🤖 Emotion detection | Multi-label emotion classification using RoBERTa (GoEmotions) |
| 📊 Psychological feature engineering | Converts raw emotion scores into interpretable psychological metrics |
| 🧠 LLM-based interpretation | Generates supportive, context-aware feedback using Qwen |
| 📈 Emotion statistics dashboard | Visualizes emotion trends and psychological indicators |
| 🔒 Non-diagnostic design | Provides supportive insights, not clinical diagnoses |

---

## 🏗️ Project Architecture

```
Journal Entry
      │
      ▼
Text Preprocessing
(Language Detection, Tanglish Correction, Normalization)
      │
      ▼
RoBERTa – GoEmotions
      │
      ▼
Emotion Probability Vector
      │
      ▼
Psychological Feature Engineering
      ├── Emotional Intensity
      ├── Emotional Diversity
      ├── Emotional Valence
      ├── Positive Affect
      ├── Negative Affect
      ├── Ambivalence
      └── Psychological Signals
      │
      ▼
Psychological Summary
      │
      ▼
Qwen LLM Interpretation
      │
      ▼
User Dashboard
```

---

## 📂 Project Structure

```
MindAura/
├── ai/
│   ├── preprocessing/     # Language detection, Tanglish correction, normalization
│   ├── training/          # Model training scripts
│   └── inference/         # Emotion detection & feature engineering inference
│
├── backend/                # FastAPI application
├── frontend/                # React application
├── docs/                    # Documentation
└── deployment/              # Docker & deployment configs
```

---

## 🛠️ Tech Stack

**AI / ML**
- Python
- Hugging Face Transformers
- RoBERTa (GoEmotions)
- Ollama
- Qwen3 14B

**Backend**
- FastAPI
- Python

**Frontend**
- React
- JavaScript

**Deployment**
- Docker
- Docker Compose

---

## 🧠 AI Pipeline

### 1. Text Preprocessing
- Language detection
- Tanglish correction
- Text normalization
- Named entity protection

### 2. Emotion Detection
RoBERTa-GoEmotions predicts probabilities across multiple emotions, including:

`Joy` · `Sadness` · `Fear` · `Anger` · `Nervousness` · `Gratitude` · `Love` · `Optimism` · `Disappointment` · `Remorse` · and others

### 3. Psychological Feature Engineering
The emotion probability vector is transformed into interpretable psychological metrics:

- Emotional Intensity
- Emotional Diversity
- Emotional Valence
- Positive Affect
- Negative Affect
- Ambivalence

**Rule-based psychological signals:**

- Mental Fatigue
- Cognitive Overload
- Restlessness
- Emotional Conflict
- Self-Criticism
- Social Withdrawal
- Helplessness Language
- Motivation Reduction

### 4. LLM Interpretation
Engineered psychological features are supplied as structured context to Qwen, which generates:

- Emotion summary
- Psychological interpretation
- Supportive reflections
- Wellness recommendations

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd MindAura
```

### 2. Backend setup
```bash
pip install -r backend/requirements.txt
```

### 3. Frontend setup
```bash
cd frontend
npm install
```

### 4. Install Ollama
Download from [ollama.com](https://ollama.com), then pull the required model:
```bash
ollama pull qwen3:14b
```

---

## ▶️ Running

**Backend**
```bash
uvicorn backend.app.main:app --reload
```

**Frontend**
```bash
npm start
```

---

## 📌 Current Progress

| Task | Status |
|---|---|
| Tanglish preprocessing | ✅ Done |
| Text normalization | ✅ Done |
| Emotion detection pipeline | ✅ Done |
| Rule-based psychological signals | ✅ Done |
| Psychological feature engineering | ✅ Done |
| Qwen interpretation refinement | 🚧 In Progress |
| Frontend integration | 🚧 In Progress |

---

## 👥 Contributors

- Sindhuja Sankaramoorthy
- Vishal Dharsan

