import pandas as pd
import os

# Define file paths
RAW_DATA_PATH = "data/raw/student_data.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_student_data.csv"

def load_data():
    """Loads raw student data from CSV."""
    if not os.path.exists(RAW_DATA_PATH):
        print("❌ Data file not found!")
        return None
    
    df = pd.read_csv(RAW_DATA_PATH)
    print("✅ Data loaded successfully!")
    
    return df

def clean_data(df):
    """Cleans and preprocesses the data."""
    print("\n🔍 Checking for missing values...")
    print(df.isnull().sum())  # Check for missing values
    
    df = df.dropna()  # Remove rows with missing values

    print("\n✅ Data cleaning complete!")
    return df

def save_data(df):
    """Saves cleaned data to processed folder."""
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)  # Ensure directory exists
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"📂 Cleaned data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = clean_data(df)
        save_data(df)
 
