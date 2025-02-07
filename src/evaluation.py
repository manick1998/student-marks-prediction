import pandas as pd
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the processed data
data_path = 'data/processed/cleaned_student_data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Processed data not found! Run feature_engineering.py first.")

df = pd.read_csv(data_path)

# Load the trained model
model_path = 'models/model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Model not found! Train the model first using training.py.")

model = joblib.load(model_path)

# Define Features and Target
X = df[['Hours_Studied']]
y = df['Total_Score']

# Make Predictions
y_pred = model.predict(X)

# Evaluate the Model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print Evaluation Metrics
print("📊 Model Evaluation Results:")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save the evaluation results
eval_results_path = 'models/evaluation_results.txt'
with open(eval_results_path, 'w') as f:
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
    f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
    f.write(f"R² Score: {r2:.2f}\n")

print(f"💾 Evaluation results saved at: {eval_results_path}")

