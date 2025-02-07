import pandas as pd
import os

def preprocess_data(df):
    """Clean and preprocess student data."""
    # Debugging: Show available columns
    print(f"📌 Available columns: {df.columns.tolist()}")

    # Ensure 'Hours_Studied' column exists
    if 'Hours_Studied' not in df.columns:
        raise KeyError("❌ Column 'Hours_Studied' not found in dataset!")

    # Drop missing values
    df = df.dropna()

    # Feature Engineering: Convert study hours into categories
    df['study_category'] = pd.cut(df['Hours_Studied'], 
                                  bins=[0, 2, 5, 8, 10], 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

def save_processed_data(df):
    """Save the cleaned data."""
    processed_path = 'data/processed/cleaned_student_data.csv'
    df.to_csv(processed_path, index=False)
    print(f'✅ Cleaned data saved to {processed_path}')

if __name__ == '__main__':
    raw_data_path = 'data/raw/student_data.csv'
    
    if not os.path.exists(raw_data_path):
        print('❌ Raw data file not found!')
    else:
        df = pd.read_csv(raw_data_path)
        df = preprocess_data(df)
        save_processed_data(df)
