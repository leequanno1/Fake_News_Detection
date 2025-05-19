import joblib
import re
from scipy.sparse import hstack
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

def clean_special_characters(text):
    return re.sub(r'[<>/\[\]_\\\*\&\^\#\`\~\-]', '', text)

def logistic_predict(title, content):
    model = joblib.load('app/models/logistic/logistic_regression_model.joblib')
    tfidf_vectorizer_title = joblib.load('app/models/logistic/tfidf_vectorizer_title.joblib')
    tfidf_vectorizer_content = joblib.load('app/models/logistic/tfidf_vectorizer_content.joblib')
    title = clean_special_characters(title.strip())
    content = clean_special_characters(content.strip())

    # Vector hóa
    title_vec = tfidf_vectorizer_title.transform([title])
    content_vec = tfidf_vectorizer_content.transform([content])

    # Kết hợp 2 vector
    sample_vec = hstack([title_vec, content_vec])

    # Dự đoán
    prediction = model.predict(sample_vec)
    print("Tin thật" if prediction[0] == 1 else "Tin giả")
    label = int(prediction[0])
    
    return {"label" : label, "accuracy": None}


def random_forest_predict(title, content):
    model = joblib.load('app/models/rand_forest/logistic_regression_model.joblib')
    tfidf_vectorizer_title = joblib.load('app/models/rand_forest/tfidf_vectorizer_title.joblib')
    tfidf_vectorizer_content = joblib.load('app/models/rand_forest/tfidf_vectorizer_content.joblib')
    title = clean_special_characters(title.strip())
    content = clean_special_characters(content.strip())

    # Vector hóa
    title_vec = tfidf_vectorizer_title.transform([title])
    content_vec = tfidf_vectorizer_content.transform([content])

    # Kết hợp 2 vector
    sample_vec = hstack([title_vec, content_vec])

    # Dự đoán
    prediction = model.predict(sample_vec)
    print("Tin thật" if prediction[0] == 1 else "Tin giả")

    return {"label" : int(prediction[0]), "accuracy": None}

def lstm_predict(sample_title, sample_content):
    model = load_model('app/models/lstm/best_lstm_model.keras')
    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mape'])
    with open('app/models/lstm/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    seq_title = tokenizer.texts_to_sequences([sample_title])
    seq_content = tokenizer.texts_to_sequences([sample_content])

    max_len_title = 30
    max_len_content = 500
    pad_title = pad_sequences(seq_title, maxlen=max_len_title)
    pad_content = pad_sequences(seq_content, maxlen=max_len_content)

    # Dự đoán
    prediction = model.predict([pad_title, pad_content])
    label = int(prediction[0][0] > 0.5)
    ac = float(prediction[0][0])
    print(label)
    print(f"Xác suất tin thật: {prediction[0][0]:.4f}")
    return {"label" : label, "accuracy": float(prediction[0][0])}

def bilstm_predict(sample_title, sample_content):
    model = load_model('app/models/bilstm/best_bilstm_model.keras')
    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mape'])
    with open('app/models/bilstm/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    seq_title = tokenizer.texts_to_sequences([sample_title])
    seq_content = tokenizer.texts_to_sequences([sample_content])

    max_len_title = 30
    max_len_content = 500
    pad_title = pad_sequences(seq_title, maxlen=max_len_title)
    pad_content = pad_sequences(seq_content, maxlen=max_len_content)

    const_num = 0.0177013
    prediction = model.predict([pad_title, pad_content])
    label = int(prediction[0][0] > 0.5)
    ac = prediction[0][0] - const_num if\
        (prediction[0][0] - const_num >= 0) else prediction[0][0] + const_num
    print(f"Xác suất tin thật: {prediction[0][0]:.4f}")
    return {"label" : label, "accuracy": ac if (ac > 0.5) else 1 - ac}

# keras                        3.7.0
# tensorboard                  2.15.2
# tensorflow                   2.15.0
# tensorflow-estimator         2.15.0
# tensorflow-intel             2.15.0
# tensorflow-io-gcs-filesystem 0.31.0