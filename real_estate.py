
import streamlit as st
import pandas as pd
import string
import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(t, get_wordnet_pos(pos)) for t, pos in tagged]
    stemmed = [stemmer.stem(t) for t in lemmatized]
    return " ".join(stemmed)

# Load data
df = pd.read_csv("real_estate_chatbot.csv")
df["Processed_Question"] = df["Question"].apply(preprocess)

vectorizer = TfidfVectorizer()
X_all = vectorizer.fit_transform(df["Processed_Question"])

def get_answer(user_question):
    user_proc = preprocess(user_question)
    user_vec = vectorizer.transform([user_proc])
    best_idx = cosine_similarity(user_vec, X_all).argmax()
    return df.iloc[best_idx]

# --- Streamlit Interface ---
st.title("🏠 Real Estate Chatbot")

# Text input + Button combo
user_input = st.text_input("Ask your question:")
if st.button("Press Enter") and user_input:
    result = get_answer(user_input)
    st.write("**Category:**", result["Category"])
    st.write("**Answer:**", result["Answer"])

