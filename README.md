
## 📩 Spam SMS/Email Classifier
[🌐Visit Website](https://spam-sms-email-classifier.streamlit.app/)


A simple web application built using **Streamlit** that detects whether a given SMS or email text is **SPAM** or **Not SPAM** using a **Multinomial Naive Bayes (MNB)** model and a **TF-IDF vectorizer**.

---

### 🔍 Features

* Input any SMS or email text.
* Instantly classify it as **SPAM** or **Not SPAM**.
* Clean, user-friendly interface.
* Efficient classification using pre-trained machine learning models.

---

### 🛠️ Tech Stack

* **Python 3**
* **Streamlit** for UI
* **Scikit-learn** for model training (offline)
* **Pickle** for model serialization
* **TF-IDF Vectorizer** for text vectorization
* **Custom preprocessing** using `utils.text_transformer`

---

### 📁 Project Structure

```
.
├── app.py                  # Main Streamlit app
├── utils.py                # Contains text preprocessing logic
├── data/
│   ├── mnb_model.pkl       # Pre-trained Multinomial Naive Bayes model
│   └── tfidf.pkl           # Trained TF-IDF vectorizer
```


### 📦 Sample Spam Messages for Testing

```text
Congratulations! You've won a $1000 Walmart gift card! Click here to claim: http://spamlink.com

URGENT: Your bank account has been locked. Click here to verify your identity.

You have been selected for a free iPhone. Reply YES to claim.
```

---

### ✍️ Author

Partho Sarothi Das
📍 Dhaka, Bangladesh
Aspiring Data Scientist | Building practical ML projects

---
