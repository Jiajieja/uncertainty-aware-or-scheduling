# [RF Quantile] Ingesting synchronized contract datasets...
# [Feature Contract] Building unified pre-scheduling feature whitelist...
# [Leakage Audit] PASS - no forbidden predictors in model feature contract.
# [Feature Contract] Total predictors: 1587
# [Feature Contract] Procedure indicators: 1572
# [Feature Contract] Train shape: (41983, 1587)
# [Feature Contract] Test shape: (17083, 1587)
# [RF Uncertainty] Fitting Random Forest over log-transformed surgical duration...
# [RF Uncertainty] Extracting predictions from 300 individual trees...
# [RF Uncertainty] Computing empirical P10/P50/P90 across tree predictions...
# [RF Uncertainty] PASS - empirical tree-prediction quantiles satisfy P10 <= P50 <= P90.
# [RF Uncertainty] Evaluating empirical tree-dispersion intervals on the temporally held-out test cohort...

# ===========================================================================
#  RANDOM FOREST UNCERTAINTY EVALUATION REPORT
#  TEMPORALLY HELD-OUT TEST COHORT: 2022-01-01 TO 2023-08-10
# ===========================================================================
#    Reference P10 Lower Bound Coverage (True >= P10)   : 90.00%
#    Empirical P10 Lower Bound Coverage                  : 60.58%
# ---------------------------------------------------------------------------
#    P50 Mean Absolute Error (minutes)                   : 93.90
# ---------------------------------------------------------------------------
#    Reference P90 Upper Bound Coverage (True <= P90)   : 90.00%
#    Empirical P90 Upper Bound Coverage                  : 59.14%
# ---------------------------------------------------------------------------
#    Reference Interquantile Coverage [P10, P90]        : 80.00%
#    Empirical Interquantile Coverage                    : 19.72%
# ===========================================================================

# [RF Uncertainty] Synchronizing predictive decision assets to optimization data layer...
# SUCCESS! RF decision artifacts generated: Prescriptive/RF_prescriptive_optimizer_inputs.csv
# Array shape: (17083, 4) (17083 temporally held-out cases).
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. DATA INGESTION AND COHORT LOADING
# ==========================================
train_input = "new_data/mover_epic_final_train_features.csv"
test_input = "new_data/mover_epic_final_test_features.csv"

print("[RF Quantile] Ingesting synchronized contract datasets...")
if not os.path.exists(train_input) or not os.path.exists(test_input):
    raise FileNotFoundError("Missing essential upstream processed cohort files.")

train_df = pd.read_csv(train_input, encoding='utf-8')
test_df = pd.read_csv(test_input, encoding='utf-8')

# ==========================================
# 2. LEAKAGE-FREE FEATURE CONTRACT
# ==========================================

print("[Feature Contract] Building unified pre-scheduling feature whitelist...")

scaler_features = [
    'AGE',
    'HEIGHT',
    'WEIGHT',
    'SURGERY_DAY_OF_WEEK',
    'SURGERY_MONTH'
]

cyclical_features = [
    'DOW_SIN',
    'DOW_COS',
    'MONTH_SIN',
    'MONTH_COS'
]

binary_features = [
    'SEX_CODE',
    'Is_Hypertension',
    'Is_Diabetes',
    'Is_Cardiac'
]

categorical_features = [
    'ASA_RATING_C'
]

procedure_cols = sorted([
    c for c in train_df.columns
    if c.startswith('PRIMARY_PROCEDURE_NM_')
])

interaction_cols = [
    'INTERACT_AGE_ASA'
]

feature_cols = sorted(
    scaler_features
    + cyclical_features
    + binary_features
    + categorical_features
    + procedure_cols
    + interaction_cols
)

# ==========================================
# DEFENSIVE LEAKAGE AUDIT
# ==========================================

forbidden_exact_features = {
    'SCHEDULED_START_HOUR',
    'IS_MORNING_CASE',
    'IS_MORNING_CASE_CODE',
    'HOUR_SIN',
    'HOUR_COS',
    'ICU_ADMIN_FLAG',
    'ICU_ADMIN_FLAG_CODE',
    'DISCH_DISP',
    'AN_START_DATETIME',
    'AN_STOP_DATETIME',
    'WHEELS_IN',
    'WHEELS_OUT',
    'IN_OR_DTTM',
    'OUT_OR_DTTM',
    'INTERACT_HOUR_MORNING'
}

forbidden_prefixes = (
    'INTERACT_ICU_x_',
)

detected_forbidden = [
    col for col in feature_cols
    if (
        col in forbidden_exact_features
        or col.startswith(forbidden_prefixes)
    )
]

assert not detected_forbidden, (
    "Leakage-control failure. Forbidden predictors detected: "
    f"{detected_forbidden}"
)

print(
    "[Leakage Audit] PASS - no forbidden predictors "
    "in model feature contract."
)

# ==========================================
# PROCEDURE SCHEMA VALIDATION
# ==========================================

test_procedure_cols = sorted([
    c for c in test_df.columns
    if c.startswith('PRIMARY_PROCEDURE_NM_')
])

assert procedure_cols == test_procedure_cols, (
    "Procedure one-hot schema mismatch between train and test cohorts."
)

# ==========================================
# FEATURE SCHEMA VALIDATION
# ==========================================

missing_train = sorted(
    set(feature_cols) - set(train_df.columns)
)

missing_test = sorted(
    set(feature_cols) - set(test_df.columns)
)

if missing_train:
    raise ValueError(
        f"Training cohort missing required features: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Test cohort missing required features: {missing_test}"
    )

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

y_train_raw = train_df['ACTUAL_DURATION'].copy()
y_train_log = np.log1p(y_train_raw)
y_test_true = test_df['ACTUAL_DURATION'].copy()

print(f"[Feature Contract] Total predictors: {len(feature_cols)}")
print(f"[Feature Contract] Procedure indicators: {len(procedure_cols)}")
print(f"[Feature Contract] Train shape: {X_train.shape}")
print(f"[Feature Contract] Test shape: {X_test.shape}")

# ==========================================
# 3. TRAIN-FITTED PREPROCESSOR PIPELINE
# ==========================================
continuous_features = (
    scaler_features
    + cyclical_features
    + interaction_cols
)

num_pipe = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='median')
    ),
    (
        'scaler',
        StandardScaler(
            with_mean=True,
            with_std=True
        )
    )
])

bin_pipe = Pipeline([
    (
        'imputer',
        SimpleImputer(
            strategy='constant',
            fill_value=0
        )
    )
])

cat_pipe = Pipeline([
    (
        'imputer',
        SimpleImputer(
            strategy='most_frequent'
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num_flow',
            num_pipe,
            continuous_features
        ),
        (
            'bin_flow',
            bin_pipe,
            binary_features
        ),
        (
            'cat_flow',
            cat_pipe,
            categorical_features
        ),
        (
            'proc_flow',
            'passthrough',
            procedure_cols
        )
    ],
    remainder='drop'
)

# ==========================================
# 4. RANDOM FOREST ENSEMBLE-DISPERSION QUANTILE APPROXIMATION
# ==========================================
print(
    "[RF Uncertainty] Fitting Random Forest "
    "over log-transformed surgical duration..."
)

N_ESTIMATORS = 300

rf_estimator = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=20,
    min_samples_leaf=10,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

pipe_rf = Pipeline([
    ('preprocess', preprocessor),
    ('model', rf_estimator)
])

pipe_rf.fit(
    X_train,
    y_train_log
)

print(
    "[RF Uncertainty] Extracting predictions "
    "from 300 individual trees..."
)

assert len(rf_estimator.estimators_) == N_ESTIMATORS, (
    "Unexpected number of fitted RF estimators."
)

X_test_transformed = (
    pipe_rf
    .named_steps['preprocess']
    .transform(X_test)
)

all_trees_predictions = np.zeros(
    (
        X_test_transformed.shape[0],
        N_ESTIMATORS
    )
)

for idx, tree in enumerate(
    rf_estimator.estimators_
):
    all_trees_predictions[:, idx] = (
        tree.predict(X_test_transformed)
    )

# Convert individual-tree predictions
# from log-duration to minutes.
all_trees_predictions_minutes = np.expm1(
    all_trees_predictions
)

print(
    "[RF Uncertainty] Computing empirical "
    "P10/P50/P90 across tree predictions..."
)

raw_quantile_outputs = {
    'P10': np.percentile(
        all_trees_predictions_minutes,
        10,
        axis=1
    ),
    'P50': np.percentile(
        all_trees_predictions_minutes,
        50,
        axis=1
    ),
    'P90': np.percentile(
        all_trees_predictions_minutes,
        90,
        axis=1
    )
}

# ==========================================
# 5. QUANTILE ORDERING VERIFICATION
# ==========================================
p10_pred = raw_quantile_outputs['P10']
p50_pred = raw_quantile_outputs['P50']
p90_pred = raw_quantile_outputs['P90']

assert np.all(
    p10_pred <= p50_pred
), "Unexpected RF quantile ordering violation: P10 > P50."

assert np.all(
    p50_pred <= p90_pred
), "Unexpected RF quantile ordering violation: P50 > P90."

print(
    "[RF Uncertainty] PASS - empirical tree-prediction "
    "quantiles satisfy P10 <= P50 <= P90."
)

# ==========================================
# 6. TEMPORALLY HELD-OUT UNCERTAINTY EVALUATION
# ==========================================

print(
    "[RF Uncertainty] Evaluating empirical "
    "tree-dispersion intervals on the temporally "
    "held-out test cohort..."
)

true_vals = y_test_true.to_numpy(dtype=np.float64)

p10_coverage = np.mean(
    true_vals >= p10_pred
) * 100

p90_coverage = np.mean(
    true_vals <= p90_pred
) * 100

interval_coverage = np.mean(
    (true_vals >= p10_pred)
    & (true_vals <= p90_pred)
) * 100

p50_mae = mean_absolute_error(
    true_vals,
    p50_pred
)

print("\n" + "="*75)
print(" RANDOM FOREST UNCERTAINTY EVALUATION REPORT")
print(" TEMPORALLY HELD-OUT TEST COHORT: 2022-01-01 TO 2023-08-10")
print("="*75)

print(
    "   Reference P10 Lower Bound Coverage (True >= P10)   : "
    "90.00%"
)
print(
    f"   Empirical P10 Lower Bound Coverage                  : "
    f"{p10_coverage:.2f}%"
)

print("-"*75)

print(
    f"   P50 Mean Absolute Error (minutes)                   : "
    f"{p50_mae:.2f}"
)

print("-"*75)

print(
    "   Reference P90 Upper Bound Coverage (True <= P90)   : "
    "90.00%"
)
print(
    f"   Empirical P90 Upper Bound Coverage                  : "
    f"{p90_coverage:.2f}%"
)

print("-"*75)

print(
    "   Reference Interquantile Coverage [P10, P90]        : "
    "80.00%"
)
print(
    f"   Empirical Interquantile Coverage                    : "
    f"{interval_coverage:.2f}%"
)

print("="*75 + "\n")

# ==========================================
# 7. EXPORT PREDICTIVE QUANTILES FOR DOWNSTREAM MILP
# ==========================================

print(
    "[RF Uncertainty] Synchronizing predictive "
    "decision assets to optimization data layer..."
)

optimizer_payload = pd.DataFrame({
    'LOG_ID': test_df['LOG_ID'].values,
    'DURATION_P10_MINS': p10_pred,
    'DURATION_P50_MINS': p50_pred,
    'DURATION_P90_MINS': p90_pred
})

# Defensive integrity checks
assert len(optimizer_payload) == len(test_df)

quantile_cols = [
    'DURATION_P10_MINS',
    'DURATION_P50_MINS',
    'DURATION_P90_MINS'
]

assert not optimizer_payload[
    quantile_cols
].isna().any().any(), (
    "NaN detected in RF optimizer predictions."
)

assert np.all(
    optimizer_payload['DURATION_P10_MINS']
    <= optimizer_payload['DURATION_P50_MINS']
)

assert np.all(
    optimizer_payload['DURATION_P50_MINS']
    <= optimizer_payload['DURATION_P90_MINS']
)

os.makedirs(
    "Prescriptive",
    exist_ok=True
)

output_optimizer_file = (
    "Prescriptive/RF_prescriptive_optimizer_inputs.csv"
)

optimizer_payload.to_csv(
    output_optimizer_file,
    index=False,
    encoding='utf-8'
)

print(
    f"SUCCESS! RF decision artifacts generated: "
    f"{output_optimizer_file}"
)

print(
    f"Array shape: {optimizer_payload.shape} "
    f"({len(optimizer_payload)} temporally held-out cases)."
)