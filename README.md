
## 📩 Email Spam Classifier

This project is a **Streamlit-based web app** that classifies email messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and a Logistic Regression model trained on TF-IDF features.

---

### 🚀 Features

- 🔍 Preprocesses email text (lowercasing, punctuation & stopword removal)
- 🧠 Predicts whether the email is spam using a trained logistic regression model
- 📊 Utilizes TF-IDF vectorization for text representation
- 🧪 **Evaluated multiple classification models using the One-vs-Rest (OvR) strategy**
- 🖥️ Built with an interactive Streamlit UI

---

### 📁 Project Structure

| File | Description |
|------|-------------|
| `spamapp.py` | Streamlit app file for live predictions |
| `Email_Spam_Classifier.ipynb` | Jupyter notebook for model training and analysis |
| `email.xls` / `email.csv` | Dataset of labeled email messages (spam/ham) |
| `spam.joblib` | Serialized trained spam classification model |
| `tfidf.joblib` | Serialized TF-IDF vectorizer |

---

### 🧠 How It Works

1. The dataset is cleaned using basic NLP techniques.
2. Text data is transformed using **TF-IDF Vectorizer**.
3. A **Logistic Regression** model is trained to classify spam.
4. **Other models** like SVM, etc., were also evaluated using **One-vs-Rest strategy** to compare performance.
5. The final model and vectorizer are saved using `joblib` for app integration.
6. Users can enter any email content in the Streamlit app and get instant predictions.

---

### 📦 Installation

```bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
pip install -r requirements.txt
```

---

### ▶️ Running the App

```bash
streamlit run spamapp.py
```

---

### 📝 Example

**Input:**
```
Congratulations! You've won a $1000 Walmart gift card. Click here to claim.
```

**Prediction:**
```
🚨 Spam!
```
