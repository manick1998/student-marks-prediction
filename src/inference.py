import joblib
import pandas as pd
import os

# Load the trained model
model_path = 'models/model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found! Train the model first using training.py.")

model = joblib.load(model_path)

def predict_score(hours_studied):
    """
    Predicts the student's total score based on hours studied.
    """
    input_data = pd.DataFrame({'Hours_Studied': [hours_studied]})
    predicted_score = model.predict(input_data)[0]
    return predicted_score

if __name__ == '__main__':
    while True:
        try:
            hours = float(input("⏳ Enter hours studied (or type 'exit' to quit): "))
            prediction = predict_score(hours)
            print(f"🎯 Predicted Score: {prediction:.2f}")
        except ValueError:
            print("🚫 Invalid input! Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

