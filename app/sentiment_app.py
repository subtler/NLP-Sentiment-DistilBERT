import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Title
st.title("✈️ Tweet Sentiment Classifier")
st.write("Enter a tweet about an airline or any topic, and this app will predict its sentiment using DistilBERT.")

# Load model & tokenizer from Hugging Face
@st.cache_resource
def load_model():
    try:
        # Using a model already fine-tuned for sentiment analysis
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

# Define label map (for the SST-2 dataset, we have 2 classes: negative=0, positive=1)
label_map = {0: "Negative", 1: "Positive"}

# Input
tweet = st.text_area("✏️ Enter a tweet:")

if st.button("Analyze Sentiment"):
    if tweet.strip():
        if model is not None and tokenizer is not None:
            # Add a spinner to show loading state
            with st.spinner("Analyzing sentiment..."):
                # Tokenize input text
                inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits).item()
                    
                    # Display result with emoji
                    emoji = "😞" if prediction == 0 else "😊"
                    st.success(f"{emoji} Predicted Sentiment: **{label_map[prediction]}**")
                    
                    # Show confidence scores
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    st.write(f"Confidence: {probs[0][prediction].item()*100:.1f}%")
        else:
            st.error("Model failed to load. Please check the logs for details.")
    else:
        st.warning("Please enter some text to analyze.")

# Add footer with info
st.markdown("---")
st.markdown("This app uses a DistilBERT model fine-tuned on the SST-2 dataset for sentiment analysis.")
