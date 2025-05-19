# 🧠 Vietnamese Fake News Detection Using AI

## 📘 Overview
This project aims to build an AI-powered system for **detecting fake news** written in **Vietnamese**. The system classifies news articles or social media posts as either **real** or **fake** using machine learning and deep learning models. The models were trained and evaluated on a curated Vietnamese dataset.

---

## 📂 Dataset
- **Language**: Vietnamese
- **Source**: Collected from social media, online newspapers, and forums.
- **Download link**: [Vietnamese Fake News Dataset (Google Drive)](https://drive.google.com/drive/folders/1R9u6ecppWsJmp491_JhDCObW8J_F4vns?usp=sharing)

---

## 🤖 Supported Models

| Model                | Type                   | Accuracy (Validation Set) |
|----------------------|------------------------|----------------------------|
| Logistic Regression  | Traditional ML         | 99.0%                      |
| Random Forest        | Traditional ML         | 99.0%                      |
| LSTM                 | Sequential Neural Net  | 99.2%                      |
| BiLSTM               | Bidirectional Neural Net | 99.1%                    |

---

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Key Libraries**:
  - `scikit-learn` for Logistic Regression and Random Forest
  - `TensorFlow` / `Keras` for LSTM and BiLSTM
  - `pandas`, `numpy` for data handling
  - `pyvi`, `underthesea` for Vietnamese NLP preprocessing

---

## ⚙️ Training Pipeline

1. **Text Preprocessing**  
   - Cleaning, tokenizing, and removing Vietnamese stopwords.
   
2. **Text Vectorization**  
   - TF-IDF for traditional ML models  
   - Tokenization + padding sequences for LSTM/BiLSTM
   
3. **Model Training**  
   - Train/test split  
   - Model training and validation evaluation

4. **Evaluation & Comparison**  
   - Compare models based on validation accuracy  
   - Select the most effective model for deployment

---

## 📊 Conclusion
All models achieved very high accuracy on the Vietnamese fake news validation set. Notably, the LSTM and BiLSTM models slightly outperformed traditional machine learning approaches, showing strong potential for real-world deployment in detecting misinformation on Vietnamese social media platforms.

---
