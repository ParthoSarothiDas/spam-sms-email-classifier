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