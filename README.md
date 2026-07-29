# 🧠 MindAura — AI-Powered Mental Wellness Journal

MindAura is an AI-powered mental wellness platform that analyzes users' journal entries (both text and voice) to provide psychologically informed emotional insights. The system combines multilingual text preprocessing, vocal emotion detection, psychological feature engineering, and Large Language Model (LLM) reasoning to generate supportive, context-aware feedback.

---

## Table of Contents
- [Features](#-features)
- [Architecture](#️-project-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#️-tech-stack)
- [AI Pipeline](#-ai-pipeline)
- [Voice & Emotion Fusion Module](#-voice--emotion-fusion-module)
- [Installation](#️-installation)
- [Running the App](#️-running)
- [Current Progress](#-current-progress)
- [Upcoming Features](#-upcoming-features)
- [Contributors](#-contributors)

---

## 🚀 Features

| Feature | Description |
|---|---|
| ✍️ Journal-based assessment | Analyzes free-form text journal entries for emotional wellness |
| 🎤 Voice-based assessment | Smart microphone recording with Voice Activity Detection (VAD) |
| 🌐 Tanglish → English normalization | Corrects and normalizes Tamil-English code-mixed text |
| 🤖 Emotion detection (Text) | Multi-label emotion classification using RoBERTa (GoEmotions) |
| 🗣️ Emotion detection (Voice) | Acoustic emotion prediction from speech tone using Wav2Vec2 |
| ⚙️ Emotion Fusion Engine | Fuses text and vocal emotions to detect "Acoustic Dissonance" |
| 📊 Psychological feature engineering | Converts raw emotion scores into interpretable psychological metrics |
| 🧠 LLM-based interpretation | Generates supportive, context-aware feedback using Qwen |
| 📈 Emotion statistics dashboard | Visualizes emotion trends and psychological indicators |
| 🔒 Non-diagnostic design | Provides supportive insights, not clinical diagnoses |

---

## 🏗️ Project Architecture

```text
User Input (Text or Voice)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Text Pathway]                     [Voice Pathway]
Text Preprocessing                 Smart Audio Capture
       │                                 │
       ▼                                 ▼
RoBERTa – GoEmotions               Faster-Whisper (Speech-to-Text) & Wav2Vec2 (Vocal Emotion)
       │                                 │
       └───────────────┬─────────────────┘
                       ▼
            Emotion Fusion Engine 
         (Calculates Acoustic Dissonance)
                       │
                       ▼
       Psychological Feature Engineering
                       │
                       ▼
             Qwen LLM Interpretation
                       │
                       ▼
                User Dashboard
```

---

## 📂 Project Structure

```text
MindAura/
├── ai/
│   ├── preprocessing/     # Language detection, Tanglish correction, normalization
│   ├── voice/             # Speech-to-text, acoustic emotion prediction, and fusion engine
│   ├── training/          # Model training scripts
│   └── inference/         # Emotion detection & feature engineering inference
│
├── backend/               # FastAPI application
├── frontend/              # React application
├── docs/                  # Documentation
└── deployment/            # Docker & deployment configs
```

---

## 🛠️ Tech Stack

**AI / ML (Text & Voice)**
- Python, PyTorch
- Hugging Face Transformers
- RoBERTa (GoEmotions) for Text Sentiment
- Wav2Vec2 for Vocal Emotion
- Faster-Whisper for Speech-to-Text
- Librosa, SoundDevice, SoundFile (Audio Processing)
- Ollama & Qwen3 14B

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
`Joy` · `Sadness` · `Fear` · `Anger` · `Nervousness` · `Gratitude` · `Love` · `Optimism` · `Disappointment` · `Remorse` · and others.

### 3. Psychological Feature Engineering
The emotion probability vector is transformed into interpretable psychological metrics:
- Emotional Intensity, Diversity, Valence
- Positive Affect, Negative Affect, Ambivalence

**Rule-based psychological signals:**
- Mental Fatigue, Cognitive Overload, Restlessness
- Emotional Conflict, Self-Criticism, Social Withdrawal

### 4. LLM Interpretation
Engineered psychological features are supplied as structured context to Qwen, which generates:
- Emotion summary, Psychological interpretation, Supportive reflections, Wellness recommendations.

---

## 🎤 Voice & Emotion Fusion Module

When a user submits a voice journal entry, a specialized workflow triggers:

1. **Audio Capture**: Real-time volume monitoring (VAD) records a clean `.wav` file.
2. **Transcription**: The audio is converted to highly accurate text using Faster-Whisper.
3. **Dual Analysis**:
   - The *text* is passed to the NLP model for sentiment.
   - The *audio* is passed to Wav2Vec2 to analyze the speaker's tone, pitch, and speed.
4. **Emotion Fusion**: The Emotion Fusion Engine compares both results to detect **Acoustic Dissonance** (e.g., saying happy words but sounding sad) and outputs a comprehensive Mental Health Distress Score (0-100) and Risk Level.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/sindhujasankaramoorthy/MindAura.AI.git
cd MindAura
```

### 2. Python Environment Setup
Install all requirements (includes heavy ML dependencies for the Voice Module):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
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

**To run the Full Voice Module (Interactive Terminal):**
```bash
python -m ai.voice.voice_pipeline
```

**Backend API**
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
| Text emotion detection pipeline | ✅ Done |
| Voice transcription (Whisper) | ✅ Done |
| Vocal emotion prediction (Wav2Vec2) | ✅ Done |
| Text & Voice Emotion Fusion Engine | ✅ Done |
| Rule-based psychological signals | ✅ Done |
| Psychological feature engineering | ✅ Done |
| Qwen interpretation refinement | ✅ Done |
| Frontend integration | 🚧 In Progress |

---

## 🔭 Upcoming Features

- **Custom Speech Emotion Recognition (SER) Model**
  - Fine-tune your own SER model.
  - Evaluate it properly.
  - Deploy it as a FastAPI endpoint.
  - Integrate it into MindAura.

---

## 👥 Contributors
- Sindhuja Sankaramoorthy
- Vishal Dharsan
