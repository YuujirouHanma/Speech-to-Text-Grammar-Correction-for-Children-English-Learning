# 🎤 Speech-to-Text & Grammar Correction for Children English Learning

## 📌 Overview

This project implements an **AI-based monitoring system for children’s English language learning** by integrating:

* Speech-to-Text (STT)
* Automated Grammar Correction (GEC)
* Transformer-based NLP models

The system converts children's speech into text and automatically corrects grammatical errors to provide **real-time learning feedback**.

---

## 🎯 Objectives

* Build an integrated pipeline for **speech transcription + grammar correction**
* Compare **Custom Seq2Seq Transformer vs Pretrained T5**
* Evaluate performance using **BLEU, METEOR, ROUGE, Exact Match**
* Support **real-time English learning monitoring for children**

---

## 🧠 System Architecture

Pipeline:

```
Audio Input → Speech-to-Text → Text Preprocessing → Grammar Correction → Output Display
```

Main Components:

* Audio Recording / Upload
* STT Engine (Whisper / Gemini / Deepgram)
* Grammar Correction Model (T5 / Custom Seq2Seq)
* Gradio Web Interface

---

## 🚀 Features

✅ Real-time speech transcription
✅ Automatic grammar correction
✅ Multi STT Engine support
✅ Custom Transformer model implementation
✅ Pretrained T5 Fine-tuning
✅ Interactive Web UI (Gradio)
✅ Multi-scenario training pipeline
✅ Educational feedback potential

---

## 🛠 Tech Stack

### AI / NLP

* T5 (Text-to-Text Transfer Transformer)
* Custom Seq2Seq Transformer (PyTorch)
* SentencePiece Tokenizer

### Speech AI

* Whisper (Local STT)
* Google Gemini Flash 2.0 / 1.5
* Deepgram Nova-2

### ML / Data

* Scikit-learn
* LightGBM
* NL-Augmenter
* ERRANT (Error Annotation Toolkit)

### Framework & Tools

* PyTorch
* HuggingFace Transformers
* HappyTransformer
* Gradio

---

## 📊 Dataset

| Dataset                    | Source      | Usage                        |
| -------------------------- | ----------- | ---------------------------- |
| Grammar Correction Dataset | Kaggle      | Supervised training          |
| JFLEG                      | HuggingFace | Fluency correction benchmark |

---

## 🧪 Experiment Scenarios

1️⃣ Auto Pseudo Labeling (ERRANT)
2️⃣ Pseudo Labeling + Augmentation
3️⃣ Pure Supervised Training
4️⃣ Direct Training on JFLEG
5️⃣ Ensemble Pseudo Labeling + Heavy Augmentation

---

## 📈 Evaluation Metrics

* BLEU
* METEOR
* ROUGE-1 / ROUGE-2 / ROUGE-L
* Exact Match
* Precision / Recall / F1 (Error Type Classification)

---

## 🏆 Key Results

### Best Performance

Model: **T5 + Pseudo Labeling + Augmentation**

| Metric      | Score  |
| ----------- | ------ |
| BLEU        | 0.8567 |
| METEOR      | 0.9329 |
| ROUGE-L     | 0.9704 |
| Exact Match | 0.6800 |

📌 T5 consistently outperformed Custom Seq2Seq across all scenarios.

---

## 🎤 Speech-to-Text Performance Insights

### Best Conditions

* Quiet environment → Highest transcription accuracy

### Challenges

* Noise environment → STT semantic errors propagate to GEC stage

### Best STT Overall

* Gemini Flash 2.0 (Balanced speed & accuracy)
* Whisper (Good local baseline)

---

## 💡 Educational Impact

This system can:

* Help children practice English speaking
* Provide instant grammar feedback
* Support teachers in monitoring student progress
* Reduce manual correction workload

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## ▶ Usage

Run Gradio interface:

```bash
python app.py
```

Then open browser → Local Gradio URL

---

## 📂 Project Structure (Example)

```
project/
│
├ models/
├ data/
├ src/
├ notebooks/
├ saved_model/
│
├ app.py
├ train.py
├ requirements.txt
└ README.md
```

---

## 🔬 Future Work

* Fine-tune STT using children speech dataset
* Upgrade to larger LLM-based GEC models
* Add educational feedback explanation system
* Real classroom user testing

---

## 👨‍💻 Author

Your Name

---

