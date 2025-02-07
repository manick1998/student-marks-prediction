import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the processed data
data_path = 'data/processed/cleaned_student_data.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Processed data not found! Run feature_engineering.py first.")

df = pd.read_csv(data_path)

# Define Features and Target
X = df[['Hours_Studied']]
y = df['Total_Score']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Training Completed!")
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📊 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save the Model
model_path = 'models/model.pkl'
joblib.dump(model, model_path)
print(f"💾 Model saved at: {model_path}")
 
