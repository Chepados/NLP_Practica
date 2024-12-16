#Importamos las librerias necesarias
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import emoji
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


#Modulo 1
def preprocess_post(text: str, root="lemmatize") -> str:

    # Convertimos a minúsculas
    text = str(text).lower()

    # Convertimos emojis a texto
    text = emoji.demojize(text)

    # Expandimos contracciones
    text = contractions.fix(text)

    # Eliminamos URLs
    text = re.sub(r'((http|https)\:\/\/)?([a-zA-Z0-9\.\-]+)\.([a-zA-Z]{2,})(\/[a-zA-Z0-9\#\?\&\=\.\_\-]*)*', '', text)

    # Eliminamos caracteres especiales y dígitos
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenizamos y eliminamos stopwords
    words = word_tokenize(text.strip())

    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]

    # Lemmatizamos, stemmizamos o no hacemos nada dependiendo del valor de root

    if root == "lemmatize":
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    elif root == "stem":
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    elif root == None:
        pass
    else:
        raise ValueError("root debe ser 'lemmatize', 'stem' o None")
    
    return " ". join([word for word in words if  21 > len(word) > 1])



#Modulo 2
def classify_subreddit(text:str):
    tfidf = TfidfVectorizer(max_features=10000)
    X_tfidf = tfidf.fit_transform(df["post_to_analize"])
    encoder = LabelEncoder()
    Y = encoder.fit_transform(df.subreddit)
    model = LogisticRegression(max_iter=5000, n_jobs=-1)
    model.fit(X_tfidf, Y)
    X_test = tfidf.transform([text])
    Y_pred = model.predict(X_test)
    return encoder.classes_[Y_pred][0]


#Modulo 3
def find_subreddit_mentions(text: str):
    re_subreddits = re.compile("|".join(list(map(lambda x: x.lower(), df.subreddit.unique()))))
    return " ".join(re_subreddits.findall(str(text).lower()))


def find_subreddit_mentions(text: str):
    re_subreddits = re.compile("|".join(list(map(lambda x: x.lower(), df.subreddit.unique()))))
    return " ".join(re_subreddits.findall(str(text).lower()))


def phone_number_extracion(text: str):
    re_phone_number = re.compile(r'(?:\+\d{1,3}[\s\-]?)?[0-9]{3}[\s\-]?\d{3}[\s\-]?\d{4}')
    return re_phone_number.findall(str(text).lower())

def code_extraction(text:str):
    re_code = re.compile(r'```(.*?)```')
    return re_code.findall(str(text))


#Modulo 4
def post_summarisation(text: str):
    """Function to summarize a text"""

    raw_sentences = nltk.sent_tokenize(text)
    procesed_sentences, cleaned_text = preprocess_text(text)

    # Utilizamos un Countvectorizer con las stopwords del texto completo preprocesado
    stop_words = stopwords.words('english')
    cv = CountVectorizer(stop_words=stop_words)
    cv_matrix = cv.fit_transform([cleaned_text])

    # Obtner las palabras únicas del texto
    unique_words = list(cv.vocabulary_.keys())
    # Creamos un diccionario de palabras y sus frecuencias
    word_freq = dict(zip(unique_words, cv_matrix.sum(axis=0).A1))

    # Calculamos la frecuencia de palabras en cada oración
    sentence_scores = {}
    for i, sentence in enumerate(procesed_sentences):
        for word, freq in word_freq.items():
            if word in sentence and word not in lista_palaras_comunes:
                raw_sentence = raw_sentences[i] if i < len(raw_sentences) else ""
                if raw_sentence in sentence_scores.keys():
                    sentence_scores[raw_sentence] += freq
                else:
                    sentence_scores[raw_sentence] = freq

    puntuaciones = np.array(list(sentence_scores.values()))
    threshold = np.quantile(puntuaciones, 0.75)
    indices = np.where(puntuaciones > threshold)[0]
    slected_raw_sentences = [sent for i, sent in enumerate(raw_sentences) if i in indices]

    sumary = "".join(slected_raw_sentences)

    return sumary


#Modulo 5

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


def obtener_embedding(texto):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    tokens = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**tokens)
    
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def texts_distance(texto1, texto2):
    embedding1 = obtener_embedding(texto1)
    embedding2 = obtener_embedding(texto2)
    
    similitud = cosine_similarity(embedding1, embedding2)
    return similitud[0][0]