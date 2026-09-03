import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 1. DATA INGESTION AND COHORT LOADING
# ==========================================
input_filename = "new_data/mover_epic_sanitized_features.csv"
print(f"[Step 1/6] Loading: {input_filename}")

if not os.path.exists(input_filename):
    raise FileNotFoundError(input_filename)

df = pd.read_csv(input_filename, encoding='utf-8')

# ==========================================
# 2. TIMESTAMP PARSING
# ==========================================
print("[Step 2/6] Parsing timestamps...")
df['IN_OR_DTTM'] = pd.to_datetime(
    df['IN_OR_DTTM'],
    errors='coerce'
)

df['OUT_OR_DTTM'] = pd.to_datetime(
    df['OUT_OR_DTTM'],
    errors='coerce'
)

df = df.dropna(
    subset=['IN_OR_DTTM', 'OUT_OR_DTTM']
).copy()

# ==========================================
# 3. AGE LOGIC
# ==========================================
df['AGE'] = pd.to_numeric(
    df['AGE'],
    errors='coerce'
)

# Remove implausible observed ages while preserving missing values
df = df[
    df['AGE'].isna()
    | df['AGE'].between(0, 110)
].copy()

# ==========================================
# 4. TARGET ENGINEERING
# ==========================================
df['ACTUAL_DURATION'] = (df['OUT_OR_DTTM'] - df['IN_OR_DTTM']).dt.total_seconds() / 60.0
df = df[(df['ACTUAL_DURATION'] > 5.0) & (df['ACTUAL_DURATION'] < 1440.0)].copy()

# ==========================================
# 5. TEMPORAL CYCLICAL FEATURES
# ==========================================
print("[Step 3/6] Creating temporal features...")
df['SURGERY_DATE'] = pd.to_datetime(
    df['SURGERY_DATE'],
    errors='coerce'
)
df = df.dropna(
    subset=['SURGERY_DATE']
).copy()
df['SURGERY_DAY_OF_WEEK'] = df['SURGERY_DATE'].dt.dayofweek
df['SURGERY_MONTH'] = df['SURGERY_DATE'].dt.month

df['DOW_SIN'] = np.sin(2 * np.pi * df['SURGERY_DAY_OF_WEEK'] / 7.0)
df['DOW_COS'] = np.cos(2 * np.pi * df['SURGERY_DAY_OF_WEEK'] / 7.0)

month_idx = df['SURGERY_MONTH'] - 1
df['MONTH_SIN'] = np.sin(2 * np.pi * month_idx / 12.0)
df['MONTH_COS'] = np.cos(2 * np.pi * month_idx / 12.0)

# ==========================================
# 6. COHORT SPLITTING
# ==========================================
print("[Step 4/6] Splitting cohorts chronologically...")
train_cohort = df[
    df['SURGERY_DATE'] < '2022-01-01'
].copy()

test_cohort = df[
    df['SURGERY_DATE'] >= '2022-01-01'
].copy()


# ==========================================
# 7. UNIFIED BINARY ENCODING
# ==========================================
print("[Step 4.1] Standardizing binary indicators...")

def standardize_binary(series):
    mapped = series.astype(str).str.strip().str.lower().map({
        'male': 1, 'female': 0,
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        '1.0': 1, '0.0': 0
    })
    return mapped.fillna(0).astype(int)


train_cohort['SEX_CODE'] = standardize_binary(train_cohort['SEX'])
test_cohort['SEX_CODE'] = standardize_binary(test_cohort['SEX'])

disease_cols = [
    'Is_Hypertension',
    'Is_Diabetes',
    'Is_Cardiac'
]

for col in disease_cols:
    train_cohort[col] = standardize_binary(train_cohort[col])
    test_cohort[col] = standardize_binary(test_cohort[col])

# ==========================================
# 8. ONE HOT ENCODING (ROBUST VERSION)
# ==========================================
print("[Step 5/6] One-hot encoding procedure names...")

train_cohort['PRIMARY_PROCEDURE_NM'] = (
    train_cohort['PRIMARY_PROCEDURE_NM']
    .fillna("UNKNOWN")
)

test_cohort['PRIMARY_PROCEDURE_NM'] = (
    test_cohort['PRIMARY_PROCEDURE_NM']
    .fillna("UNKNOWN")
)

# Fit ONLY on the training cohort
ohe = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

proc_train = np.asarray(
    ohe.fit_transform(
        train_cohort[['PRIMARY_PROCEDURE_NM']]
    )
)

# Test cohort is transformed only — no fitting
proc_test = np.asarray(
    ohe.transform(
        test_cohort[['PRIMARY_PROCEDURE_NM']]
    )
)

proc_cols = ohe.get_feature_names_out(
    ['PRIMARY_PROCEDURE_NM']
).tolist()

train_proc_df = pd.DataFrame(
    data=proc_train,
    columns=proc_cols,
    index=train_cohort.index
)

test_proc_df = pd.DataFrame(
    data=proc_test,
    columns=proc_cols,
    index=test_cohort.index
)

train_cohort = pd.concat(
    [train_cohort, train_proc_df],
    axis=1
)

test_cohort = pd.concat(
    [test_cohort, test_proc_df],
    axis=1
)

os.makedirs(
    "new_data",
    exist_ok=True
)

joblib.dump(
    ohe,
    "new_data/procedure_ohe_encoder.pkl"
)

# ==========================================
# 9. DOMAIN-DRIVEN FEATURE INTERACTIONS CROSSING (CRITICAL落盘修复)
# ==========================================
print("[Step 5.1] Engineering explicit domain-specific interaction metrics directly into storage files...")

# Ensure interaction mapping loops execute inside upstream storage phase
for df_block in [train_cohort, test_cohort]:
    # 1. Physiological Frailty Synergy Cross Term
    df_block['INTERACT_AGE_ASA'] = df_block['AGE'] * df_block['ASA_RATING_C']
    
# ==========================================
# 10. CLEANUP & SAVE
# ==========================================
print("[Step 6/6] Clearing redundant labels and saving final contract matrices...")
redundant_cols = [
    'SEX',
    'PRIMARY_PROCEDURE_NM',
    'ASA_RATING'
]
train_cohort.drop(columns=redundant_cols, errors='ignore', inplace=True)
test_cohort.drop(columns=redundant_cols, errors='ignore', inplace=True)

train_output = "new_data/mover_epic_final_train_features.csv"
test_output = "new_data/mover_epic_final_test_features.csv"

train_cohort.to_csv(train_output, index=False, encoding='utf-8')
test_cohort.to_csv(test_output, index=False, encoding='utf-8')

print(f"FEATURE ENGINEERING SYSTEM UPLOADED - Train: {train_cohort.shape} | Test: {test_cohort.shape}\n")

print(train_cohort.shape)
print(test_cohort.shape)

print(train_cohort.columns.tolist())
print(test_cohort.columns.tolist())

print(train_cohort[
    ['SURGERY_DATE', 'ACTUAL_DURATION']
].head())


forbidden_patterns = [
    'SCHEDULED_START_HOUR',
    'IS_MORNING',
    'HOUR_SIN',
    'HOUR_COS',
    'ICU',
    'DISCH',
    'AN_START',
    'AN_STOP',
    'WHEELS'
]

for pattern in forbidden_patterns:
    matches = [
        c for c in train_cohort.columns
        if pattern.upper() in c.upper()
    ]
    print(f"{pattern}: {matches}")

print("\nInteraction columns:")
print([
    c for c in train_cohort.columns
    if c.startswith('INTERACT_')
])

print("\nCohort date ranges:")
print(
    "Train:",
    train_cohort['SURGERY_DATE'].min(),
    "to",
    train_cohort['SURGERY_DATE'].max()
)
print(
    "Test:",
    test_cohort['SURGERY_DATE'].min(),
    "to",
    test_cohort['SURGERY_DATE'].max()
)

print("\nMissing values in final intended predictors:")
base_features = [
    'AGE', 'HEIGHT', 'WEIGHT',
    'SURGERY_DAY_OF_WEEK', 'SURGERY_MONTH',
    'DOW_SIN', 'DOW_COS',
    'MONTH_SIN', 'MONTH_COS',
    'SEX_CODE',
    'Is_Hypertension', 'Is_Diabetes', 'Is_Cardiac',
    'ASA_RATING_C',
    'INTERACT_AGE_ASA'
]

print(train_cohort[base_features].isna().sum())
print("\nTest:")
print(test_cohort[base_features].isna().sum())