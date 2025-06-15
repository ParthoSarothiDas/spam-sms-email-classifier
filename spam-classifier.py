import streamlit as st
import os
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

def text_transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    alnum = []
    for i in text:
        if i.isalnum():
            alnum.append(i)
            
    alnum_stop = []
    for i in alnum:
        if i not in stopwords.words('english'):
            alnum_stop.append(i)

    final_text = []
    for i in alnum_stop:
        final_text.append(ps.stem(i))
    return " ".join(final_text)

@st.cache_resource
def load_model():
    with open('data/mnb_model.pkl', 'rb') as file:
        mnb = pickle.load(file)
    return mnb

@ st.cache_resource
def load_tfidf():
    with open('data/tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    return tfidf

mnb = load_model()
tfidf = load_tfidf()


st.title('Spam SMS/Email Classifier')
text = st.text_area('Input your SMS or email for classify:')
clicked = st.button('Classify')
if clicked:
    if text == '':
        st.error('Please input sms or email for classify.')
    else:
        transformed_text = text_transformer(text)
        vectorized = tfidf.transform([transformed_text])
        result = mnb.predict(vectorized)[0]

        if result == 0:
            st.info('This is not a spam')
        else:
            st.error('This is a SPAM.')


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; font-size: 0.9em; color: gray;">
    Created by <b>Partho Sarothi Das</b><br>
    <i>Aspiring Data Scientist | Passionate about ML & Visualization</i><br>
    Email: <a href="mailto:partho52@gmail.com">partho52@gmail.com</a>
</div>
""", unsafe_allow_html=True)
