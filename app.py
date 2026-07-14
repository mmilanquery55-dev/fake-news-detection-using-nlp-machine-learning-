import streamlit as st 
import pickle 
import string
import nltk 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

def transform_content(content):
    content=content.lower()
    content=content.translate(str.maketrans('','',string.punctuation))
    tokens=word_tokenize(content)
    tokens=[word for word in tokens if word not in stop_words]
    tokens=[lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

try:
    tfidf=pickle.load(open('vectorizer.pkl','rb'))
    model=pickle.load(open('logistic_model.pkl','rb'))

except FileNotFoundError:
    st.error("Error : Model or Vector not found.")
    st.stop()

st.title("📰 Fake News Detection System")
input_msg=st.text_area("Paste or type a news article here...") 

if st.button("predict"):
    if input_msg.strip() == "":
        st.warning("Please enter a news to classify.")
    else:
        transformed_msg=transform_content(input_msg)
        vector_input=tfidf.transform([transformed_msg])
        prediction=model.predict(vector_input)[0]

        if prediction == 0:
           st.error("🚨 Prediction: Fake News")
        else:
           st.success("✅ Prediction: True News")

     

