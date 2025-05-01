import asyncio
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Title
st.title("✈️ Tweet Sentiment Classifier")
st.write("Enter a tweet about an airline or any topic, and this app will predict its sentiment using DistilBERT.")

# Load model & tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# Define label map
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Input
tweet = st.text_area("✏️ Enter a tweet:")

if st.button("Analyze Sentiment"):
    if tweet.strip():
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()
            st.success(f"🎯 Predicted Sentiment: **{label_map[prediction]}**")
    else:
        st.warning("Please enter some text to analyze.")
