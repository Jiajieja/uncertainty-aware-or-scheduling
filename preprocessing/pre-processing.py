import pandas as pd
import numpy as np
import os

# ==========================================
# 1. DATA INGESTION (Only load core structural tables)
# ==========================================
print("[Step 1/6] Ingesting core MOVER relational tables...")
df_info = pd.read_csv("original_data/patient_information.csv", encoding='utf-8')
df_history = pd.read_csv("original_data/patient_history.csv", encoding='utf-8')

print("[Age Recovery] Recovering AGE variable from de-identified BIRTH_DATE field...")

if 'BIRTH_DATE' in df_info.columns:
    
    # In this MOVER release BIRTH_DATE stores age rather than actual DOB
    df_info['AGE'] = pd.to_numeric(
        df_info['BIRTH_DATE'],
        errors='coerce'
    )
    
    print(
        f"[Age Recovery] AGE recovered successfully. "
        f"Missing AGE records: {df_info['AGE'].isna().sum()}"
    )
else:
    raise ValueError(
        "BIRTH_DATE column not found. Unable to reconstruct AGE feature."
    )

# ==========================================
# 2. COLUMN PRUNING (Select relevant features and targets)
# ==========================================
print("[Step 2/6] Extracting operational predictors and target variables...")
core_info_cols = [
    'LOG_ID',
    'MRN',
    'AGE',
    'SEX',
    'HEIGHT',
    'WEIGHT',
    'ASA_RATING_C',
    'ASA_RATING',
    'PRIMARY_PROCEDURE_NM',
    'SURGERY_DATE',
    'IN_OR_DTTM',
    'OUT_OR_DTTM'
]
df_info_core = df_info[[c for c in core_info_cols if c in df_info.columns]].copy()

# ==========================================
# 3. PATIENT HISTORY AGGREGATION (Mitigate One-to-Many explosion)
# ==========================================
print("[Step 3/6] Transforming clinical history into one-hot categorical matrices...")
df_history.rename(columns={'mrn': 'MRN'}, inplace=True)

df_history['Is_Hypertension'] = df_history['dx_name'].str.contains('hypertension|hypertensive', case=False, na=False).astype(int)
df_history['Is_Diabetes'] = df_history['dx_name'].str.contains('diabetes|diabetic', case=False, na=False).astype(int)
df_history['Is_Cardiac'] = df_history['dx_name'].str.contains('heart|cardiac', case=False, na=False).astype(int)

df_history_features = df_history.groupby('MRN')[['Is_Hypertension', 'Is_Diabetes', 'Is_Cardiac']].max().reset_index()

# ==========================================
# 4. RELATIONAL DATA INTEGRATION
# ==========================================
print("[Step 4/6] Merging patient information with pre-existing clinical history...")

final_feature_matrix = pd.merge(
    df_info_core,
    df_history_features,
    on='MRN',
    how='left'
)

disease_cols = [
    'Is_Hypertension',
    'Is_Diabetes',
    'Is_Cardiac'
]

final_feature_matrix[disease_cols] = (
    final_feature_matrix[disease_cols]
    .fillna(0)
    .astype(int)
)
# ==========================================
# 5. RAW DATA QUALITY AUDIT (Before Processing)
# ==========================================
print("[Step 5/6] Generating pre-processing data quality audit report...")
null_counts_raw = final_feature_matrix.isnull().sum()
null_percentages_raw = (final_feature_matrix.isnull().sum() / len(final_feature_matrix)) * 100
missing_report_raw = pd.DataFrame({'Missing_Counts': null_counts_raw, 'Percentage_%': null_percentages_raw.round(2)})

print("\n" + "="*50)
print("     RAW DATA QUALITY MISSINGNESS REPORT (BEFORE PROCESSING)")
print("="*50)
print(missing_report_raw)
print(f"Matrix Dimensions: {final_feature_matrix.shape[0]} rows x {final_feature_matrix.shape[1]} columns")
print("="*50 + "\n")

# ==========================================
# 6. MISSING VALUE GOVERNANCE
# ==========================================
print("[Step 6/6] Applying missing-data and data-quality rules...")


def parse_height_to_cm(x):
    if pd.isna(x):
        return np.nan

    x_str = str(x).strip()

    if "'" in x_str:
        try:
            parts = x_str.split("'")
            feet = float(parts[0].strip())
            inches_str = parts[1].replace('"', '').strip()
            inches = float(inches_str) if inches_str else 0.0
            return (feet * 30.48) + (inches * 2.54)
        except Exception:
            return np.nan

    try:
        return float(x_str)
    except ValueError:
        return np.nan


# 1. Standardize height representation
final_feature_matrix['HEIGHT'] = (
    final_feature_matrix['HEIGHT']
    .apply(parse_height_to_cm)
    .round(1)
)


# 2. Parse scheduling date
final_feature_matrix['SURGERY_DATE'] = pd.to_datetime(
    final_feature_matrix['SURGERY_DATE'],
    errors='coerce'
)


# 3. Parse actual timestamps used ONLY for target construction
final_feature_matrix['IN_OR_DTTM'] = pd.to_datetime(
    final_feature_matrix['IN_OR_DTTM'],
    errors='coerce'
)

final_feature_matrix['OUT_OR_DTTM'] = pd.to_datetime(
    final_feature_matrix['OUT_OR_DTTM'],
    errors='coerce'
)


# 4. Remove records without scheduling date or target timestamps
required_cols = [
    'SURGERY_DATE',
    'IN_OR_DTTM',
    'OUT_OR_DTTM',
    'PRIMARY_PROCEDURE_NM'
]

final_feature_matrix.dropna(
    subset=required_cols,
    inplace=True
)


# 5. Preserve missing continuous values.
# AGE, HEIGHT and WEIGHT will be imputed using training-set
# statistics inside the downstream sklearn pipeline.


# 6. Treat missing ASA rating as an explicit unknown category
final_feature_matrix['ASA_RATING_C'] = (
    pd.to_numeric(
        final_feature_matrix['ASA_RATING_C'],
        errors='coerce'
    )
    .fillna(0)
    .astype(int)
)

final_feature_matrix['ASA_RATING'] = (
    final_feature_matrix['ASA_RATING']
    .fillna('Unknown')
)
# Generate processed data quality audit report
null_counts_clean = final_feature_matrix.isnull().sum()
null_percentages_clean = (final_feature_matrix.isnull().sum() / len(final_feature_matrix)) * 100
missing_report_clean = pd.DataFrame({'Missing_Counts': null_counts_clean, 'Percentage_%': null_percentages_clean.round(2)})

print("\n" + "="*50)
print("     CLEANED DATA QUALITY MISSINGNESS REPORT (AFTER PROCESSING)")
print("="*50)
print(missing_report_clean)
print(f"Matrix Dimensions: {final_feature_matrix.shape[0]} rows x {final_feature_matrix.shape[1]} columns")
print("="*50 + "\n")
print("AGE Variable Audit")
print("-"*50)
print(f"Minimum AGE : {final_feature_matrix['AGE'].min()}")
print(f"Maximum AGE : {final_feature_matrix['AGE'].max()}")
print(f"Median AGE  : {final_feature_matrix['AGE'].median()}")
print("-"*50 + "\n")
# ==========================================
# 8. HIGH-FIDELITY PERSISTENCE (Absolute UTF-8 Serialization)
# ==========================================
print("Serializing sanitized feature matrix into absolute UTF-8 schema...")
output_filename = "new_data/mover_epic_sanitized_features.csv"

try:
    final_feature_matrix.to_csv(output_filename, index=False, encoding='utf-8')
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print("="*50)
    print("PIPELINE EXECUTION AND SERIALIZATION COMPLETED SUCCESSFULLY!")
    print(f"Saved Location : {output_filename}")
    print(f"Matrix Dimensions: {final_feature_matrix.shape[0]} rows x {final_feature_matrix.shape[1]} columns")
    print(f"Exported Size : {file_size_mb:.2f} MB")
    print("="*50)

except Exception as e:
    print(f"Error during file persistence serialization: {str(e)}")