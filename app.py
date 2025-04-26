import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)  

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and logistic model
try:
    cv = joblib.load('count_vectorizer.joblib')
    logistic_model = joblib.load('logistic_regression_model.joblib')
    st.success("Model and vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Streamlit UI
st.title("📩 SMS Spam Classifier (Logistic Regression)")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = cv.transform([transformed_sms])

        # Predict using Logistic Regression model
        result = logistic_model.predict(vector_input)[0]

        # Output
        if result == 0:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")
