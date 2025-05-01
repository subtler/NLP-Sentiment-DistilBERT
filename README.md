<<<<<<< HEAD

=======
# ✨ Sentiment Analysis using DistilBERT (Multi-Class NLP Classifier)

A transformer-based NLP project that classifies tweet sentiments into **Positive**, **Neutral**, or **Negative** using **DistilBERT** from HuggingFace. This project showcases end-to-end fine-tuning, evaluation, and interface building for real-world applications in text classification.

---

## 📌 Project Overview

- 🎯 **Goal**: Build a multi-class sentiment classifier for airline-related tweets.
- ✅ **Classes**: Positive, Neutral, Negative
- ⚙️ **Model**: Fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- 📊 **Dataset**: [Tweets.csv](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (14k+ labeled tweets)

---

## 💡 Features

- Tokenization with `DistilBertTokenizerFast`
- Fine-tuning using HuggingFace `Trainer` API
- Accuracy and F1-score evaluation
- Streamlit app for live prediction (optional)

---

## 🧠 Model Architecture

- Pretrained DistilBERT model from HuggingFace
- Classification head with 3 output neurons (softmax)
- Optimized using AdamW
- Trained over 3 epochs on Colab GPU

---

## 📦 Tech Stack

- Python
- Transformers (HuggingFace)
- PyTorch
- scikit-learn
- Streamlit (optional UI)
- Google Colab (training)

---

## 📊 Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | **81%** |
| F1-Score  | **~80%** |
| Model     | DistilBERT (fine-tuned) |

---

## 💻 How to Run

### 🔧 Setup
```bash
pip install -r requirements.txt
>>>>>>> 📝 Added professional README and structured folders
