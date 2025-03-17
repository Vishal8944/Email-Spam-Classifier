  

---

# **📩 Email Spam Classifier**  

## 🚀 **Overview**  
This project is an **Email Spam Classifier** that detects whether an email is **spam or non-spam** using **machine learning and natural language processing (NLP) techniques**. The model is trained using a dataset of emails and deployed as an **interactive web application** with **Streamlit**.  

## 🛠 **Technologies Used**  
- **Python**  
- **pandas** (for data manipulation)  
- **nltk** (for text preprocessing)  
- **scikit-learn** (for machine learning models)  
- **TF-IDF Vectorizer** (for feature extraction)  
- **Logistic Regression** (for classification)  
- **Streamlit** (for web app deployment)  
- **joblib** (for saving and loading models)  

## 📂 **Dataset**  
The project uses an **email dataset (`email.csv`)**, where each email is labeled as:  
- **Spam (1)** – Unwanted or promotional messages  
- **Not Spam (0)** – Genuine email messages  

## 🔧 **Installation & Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/email-spam-classifier.git
   cd email-spam-classifier
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the **Streamlit** app:  
   ```bash
   streamlit run spamapp.py
   ```

## 📊 **How It Works**  
### **1. Model Training (`email_spam_classifier.ipynb`)**  
- Loads and preprocesses the email dataset.  
- Uses **TF-IDF Vectorization** to convert text into numerical features.  
- Trains a **Logistic Regression model** for spam classification.  
- Saves the trained model and vectorizer using **joblib**.  

### **2. Web Application (`spamapp.py`)**  
- Loads the **trained spam classification model** and **TF-IDF vectorizer**.  
- Allows users to input an email message.  
- Preprocesses and classifies the email as **Spam** or **Not Spam**.  
- Displays results using **Streamlit** UI.  

## ✨ **Features**  
✅ **Detects spam emails** with a trained machine learning model  
✅ **Simple and interactive Streamlit UI**  
✅ **Real-time email classification**  
✅ **Uses NLP techniques for text cleaning**  

## 🖼 **Screenshots**  
| Email Input | Classification Output |
|-------------|----------------------|
| ![User Input](https://via.placeholder.com/300x150?text=Email+Input) | ![Spam Prediction](https://via.placeholder.com/300x150?text=Spam+Result) |

## 🚀 **Future Improvements**  
- Improve accuracy with **deep learning models (LSTM, Transformers)**  
- Deploy as a **web app using Flask/Django**  
- Enhance dataset with **real-world email samples**  

---
