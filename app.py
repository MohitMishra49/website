from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Spam Detection API Running"}

@app.post("/predict")
def predict(data: dict):
    message = data["message"]
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0][1]

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(float(probability)*100, 2)
    }