'''import pandas as pd

def combine_and_clean_datasets(file1_path, file2_path, output_path):
    print(f"Loading data from {file1_path} and {file2_path}...")
    
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    if 'diseases' in df1.columns:
        df1.rename(columns={'diseases': 'target_disease'}, inplace=True)
    if 'prognosis' in df2.columns:
        df2.rename(columns={'prognosis': 'target_disease'}, inplace=True)

    df1.columns = df1.columns.str.lower().str.replace(' ', '_')
    df2.columns = df2.columns.str.lower().str.replace(' ', '_')

    # --- NEW FIX: Drop duplicate columns ---
    df1 = df1.loc[:, ~df1.columns.duplicated()]
    df2 = df2.loc[:, ~df2.columns.duplicated()]
    # ---------------------------------------

    synonym_map = {
        'shortness_of_breath': 'breathlessness',
        'sharp_chest_pain': 'chest_pain',
        'stomach_bloating': 'swelling_of_stomach',
        'abdominal_pain': 'stomach_pain'
    }
    df1.rename(columns=synonym_map, inplace=True)
    df2.rename(columns=synonym_map, inplace=True)

    # We must run the duplicate drop ONE MORE TIME in case the synonym map created new duplicates
    df1 = df1.loc[:, ~df1.columns.duplicated()]
    df2 = df2.loc[:, ~df2.columns.duplicated()]

    combined_df = pd.concat([df1, df2], ignore_index=True)

    combined_df.fillna(0, inplace=True)

    combined_df.to_csv(output_path, index=False)
    print(f"Success! Combined dataset saved to: {output_path}")
    print(f"Total Rows: {combined_df.shape[0]}, Total Features: {combined_df.shape[1]}")

    return combined_df

if __name__ == "__main__":
    combine_and_clean_datasets(
        'data/raw/raw1.csv', 
        'data/raw/raw2.csv', 
        'data/raw/symptom_dataset.csv'
    )'''



import pandas as pd
import numpy as np

def clean_and_process_data(input_file, output_file):
    print(f"Loading combined data from: {input_file}")
    
    # FIX 1: Added low_memory=False to handle the DtypeWarning safely
    df = pd.read_csv(input_file, low_memory=False) 
    
    initial_rows = len(df)
    print(f"Initial dataset shape: {df.shape}")
    # FIX 2: Bulletproof Target Column Renaming
    print("Locating and standardizing the target disease column...")
    
    # Strip hidden whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    # Dynamically hunt for the disease column
    possible_targets = ['target_disease', 'diseases', 'disease', 'prognosis']
    target_col = None
    
    for col in df.columns:
        if col.lower() in possible_targets:
            target_col = col
            break
            
    if target_col is None:
        print(f"❌ ERROR: Could not find the target column! Here are the columns I see:\n{list(df.columns)[:20]}...")
        return
        
    # Rename it to our standard name
    if target_col != 'target_disease':
        df.rename(columns={target_col: 'target_disease'}, inplace=True)
    
    # Now we can safely drop rows with missing targets
    df.dropna(subset=['target_disease'], inplace=True) 
    df['target_disease'] = df['target_disease'].astype(str).str.strip().str.title()
    
    # 2. Drop non-predictive columns (Data Leakage)
    if 'medicine' in df.columns:
        df.drop(columns=['medicine'], inplace=True)
        print("Dropped 'medicine' column to prevent data leakage.")
        
    # 3. Ensure all symptom columns are strictly binary integers (0 or 1)
    print("Binarizing symptom data...")
    symptom_cols = [col for col in df.columns if col != 'target_disease']
    
    for col in symptom_cols:
        # Convert text/errors to NaN, fill NaN with 0, and force 1s and 0s
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].apply(lambda x: 1 if x > 0 else 0).astype(int)

    # 4. Drop empty columns (Symptoms that no patient in the dataset has)
    empty_cols = [col for col in symptom_cols if df[col].sum() == 0]
    df.drop(columns=empty_cols, inplace=True)
    if len(empty_cols) > 0:
        print(f"Dropped {len(empty_cols)} empty symptom columns.")

    # 5. Remove exact duplicate rows to prevent model bias
    df.drop_duplicates(inplace=True)
    final_rows = len(df)
    print(f"Dropped {initial_rows - final_rows} duplicate patient records.")

    # 6. Save the perfectly processed data
    df.to_csv(output_file, index=False)
    print(f"\n✅ SUCCESS! Processed data saved to: {output_file}")
    print(f"📊 Final Dataset Shape for Training: {df.shape[0]} rows, {df.shape[1]} columns")

if __name__ == "__main__":
    # Define input and output paths
    INPUT_PATH = 'data/raw/symptom_dataset.csv'
    OUTPUT_PATH = 'data/processed/processed_symptoms.csv'
    
    clean_and_process_data(INPUT_PATH, OUTPUT_PATH)