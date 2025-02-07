from fastapi import FastAPI
import joblib
import pandas as pd
#Fast Api

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model_path = 'models/model.pkl'
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.get("/")
def home():
    return {"message": "Welcome to Student Marks Prediction API!"}

@app.get("/predict/")
def predict(hours_studied: float):
    """
    Predicts student score based on hours studied.
    """
    input_data = pd.DataFrame({'Hours_Studied': [hours_studied]})
    predicted_score = model.predict(input_data)[0]
    return {"hours_studied": hours_studied, "predicted_score": round(predicted_score, 2)}

# Run the API: `uvicorn src.deployment:app --reload`
