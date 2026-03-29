import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. Helper Functions ---
def parse_fraction(value):
    """Converts '1/250' to 0.004."""
    if pd.isna(value) or value == "":
        return np.nan
    try:
        value = str(value)
        if '/' in value:
            num, den = value.split('/')
            return float(num) / float(den)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return np.nan

def calculate_vif(X):
    """Calculates Variance Inflation Factor for a DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    # We add a constant for the intercept as required by statsmodels
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

# --- 2. Load Data ---
# Swap this with df = pd.read_csv("your_kaggle_file.csv") 
# Creating dummy data for demonstration:
data = {
    'ExposureTime': ['1/250', '1/100', '1/500', np.nan, '1/1000'],
    'ISO': [100, 200, 100, 400, 800],
    'FNumber': [2.8, 4.0, 2.8, 5.6, 8.0],
    'CameraBrand': ['Canon', 'Nikon', 'Canon', 'Sony', 'Nikon']
}
df = pd.DataFrame(data)

# --- 3. Preprocessing Steps ---

# Step A: Parse fractions
df['ExposureTime_Num'] = df['ExposureTime'].apply(parse_fraction)

# Step B: One-Hot Encoding
# We drop the original categorical column and the messy string column
df_encoded = pd.get_dummies(df.drop(columns=['ExposureTime']), columns=['CameraBrand'], drop_first=True)

# Step C: Handle Missing Values (NaNs)
# SimpleImputer requires numerical data
imputer = SimpleImputer(strategy='mean')
df_final = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

# --- 4. Multicollinearity Check (VIF) ---
# Note: High VIF (>5 or 10) means features are redundant
print("--- Preprocessed Matrix ---")
print(df_final.head())

print("\n--- VIF Results ---")
print(calculate_vif(df_final))