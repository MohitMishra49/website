from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download once (important for Render)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


@app.post("/predict")
def predict(data: dict):
    message = data["message"]

    # 🔥 IMPORTANT — SAME preprocessing as Streamlit
    transformed_sms = transform_text(message)

    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input)[0][1]

    return {
        "prediction": "Spam" if result == 1 else "Not Spam",
        "confidence": float(probability)
    }