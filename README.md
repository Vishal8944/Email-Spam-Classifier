  

---

# 📩 **Email Spam Classifier**  
A machine learning-based spam detection system that classifies emails as **Spam or Not Spam** using **NLP and TF-IDF Vectorization**. Built with **Python, scikit-learn, and Streamlit**, this project includes a trained model and an interactive web app for real-time classification.  

## 🚀 **Project Overview**  
This project uses **Natural Language Processing (NLP)** techniques and multiple **machine learning models** to classify emails as spam or not spam. The classifier is deployed as a **Streamlit web app** for real-time predictions.  

## 🛠 **Technologies Used**  
- **Python**  
- **pandas** – Data manipulation  
- **nltk** – Text preprocessing (stopword removal, cleaning)  
- **scikit-learn** – Machine learning models  
- **TF-IDF Vectorizer** – Feature extraction  
- **Seaborn** – Data visualization  
- **Streamlit** – Web app for real-time predictions  
- **joblib** – Model serialization  

## 🔍 **Models Tested**  
The project evaluates different machine learning models for spam classification:  
✔️ **Logistic Regression**  
✔️ **Decision Tree**  
✔️ **K-Nearest Neighbors (KNN)**  

## 📂 **Dataset**  
The project uses an email dataset (`email.csv`) that contains email text labeled as **Spam (1)** or **Not Spam (0)**.  

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
3. Run the Streamlit app:  
   ```bash
   streamlit run spamapp.py
   ```

## 🏗 **How It Works**  
1. **Load and preprocess dataset**  
2. **Train multiple models (Logistic Regression, Decision Tree, KNN) using TF-IDF features**  
3. **Evaluate and compare model performance**  
4. **Save the best model (`spam_classifier.joblib`) and vectorizer (`tfidf_vectorizer.joblib`)**  
5. **Deploy an interactive web app using Streamlit**  
6. **User enters email text → Model predicts Spam/Not Spam**  

## 🎯 **Key Features**  
✅ **Real-time email classification**  
✅ **Multiple machine learning models tested**  
✅ **Text preprocessing (stopword removal, lowercasing, cleaning)**  
✅ **User-friendly web app using Streamlit**  
✅ **Data visualization with Seaborn**  

## 📊 **Results & Performance**  
Each model is evaluated using accuracy and classification reports to determine the best-performing spam classifier.  

## ✨ **Future Improvements**  
- Integrate **deep learning models (LSTMs, Transformers)** for better accuracy  
- Add support for **multiple languages**  
- Improve **spam detection with advanced NLP techniques**  

---
