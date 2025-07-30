# # you need to install all these in your terminal
# # pip install streamlit
# # pip install scikit-learn
# # pip install python-docx
# # pip install PyPDF2

import streamlit as st
import pickle
import docx
import PyPDF2
import re


# Clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Load model, vectorizer, and encoder
with open("clf.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# File text extractor
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


# Streamlit UI
st.title("Resume Screening App")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    raw_text = extract_text_from_file(uploaded_file)
    cleaned_text = cleanResume(raw_text)
    transformed_text = tfidf.transform([cleaned_text])

    prediction = model.predict(transformed_text)[0]
    predicted_category = encoder.inverse_transform([prediction])[0]

    st.subheader("Predicted Job Category:")
    st.success(predicted_category)
