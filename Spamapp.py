import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("email.csv")
nltk.download('stopwords')
def clean(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    text = " ".join(word for word in text.split() if word not in stopwords.words('english'))  # Remove stopwords
    return text
# Load the trained model
loaded_model = joblib.load("spam.joblib")

# Load the TF-IDF Vectorizer
loaded_vectorizer = joblib.load("tfidf.joblib")










st.title("📩 Email Spam Classifier")
st.write("Enter an email message below to check if it's Spam or Non-Spam.")

# User Input
user_input = st.text_area("✉️ Type or paste the email content here:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess user input
        cleaned_input = clean(user_input)
        
        # Convert input to TF-IDF format
        input_vectorized = loaded_vectorizer.transform([cleaned_input]).toarray()
        
        # Make Prediction
        prediction = loaded_model.predict(input_vectorized)[0]
        
        # Display Result
        if prediction == 1:
            st.error("🚨 This email is **Spam**!")
        else:
            st.success("✅ This email is **Not Spam**.")
