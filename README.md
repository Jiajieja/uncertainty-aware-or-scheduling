# Uncertainty-Aware Predict-then-Optimize for Operating Room Scheduling

An uncertainty-aware **Predict-then-Optimize (PTO)** framework for
surgical-duration prediction and operating-room (OR) scheduling under
duration uncertainty.

## Project Overview

Operating-room schedules are highly sensitive to uncertainty in surgical
duration. A model that performs well on conventional predictive metrics
does not necessarily produce better downstream schedules once its
predictions are embedded in an optimization model.

This repository contains the final validated research pipeline for an
uncertainty-aware PTO framework that:

1.  preprocesses retrospective surgical data and constructs
    leakage-controlled predictive features;
2.  estimates case-specific surgical-duration uncertainty using
    **LightGBM, Random Forest, and XGBoost**;
3.  propagates P50 and P90 duration estimates into a risk-aware
    mixed-integer linear programming (MILP) scheduler;
4.  evaluates planned schedules using observed surgical durations
    without hindsight re-optimization; and
5.  examines computational performance, duration-cap sensitivity, and
    cross-cohort robustness.

The central research question is not only which model predicts surgical
duration most accurately, but how predictive uncertainty propagates into
downstream scheduling decisions and realized operational outcomes.

## Framework

``` text
Retrospective surgical records
          |
          v
Data preprocessing and feature engineering
          |
          v
Surgical-duration prediction
   +-------------------------------+
   | LightGBM | Random Forest | XGBoost |
   +-------------------------------+
          |
          v
Case-specific P10 / P50 / P90 estimates
          |
          v
Risk-aware duration
d_i(lambda) = (1-lambda) P50_i + lambda P90_i
          |
          v
MILP room assignment + endogenous sequencing
          |
          v
Time-limited feasible incumbent schedule
          |
          v
Observed-duration replay
          |
          v
Overtime / makespan / propagated start delay
```

## Dataset and Data Availability

The study uses the **MOVER** surgical dataset and is based on **59,066
retrospective surgical records** after final cleaning:

  Split                                Cases
  --------------------------------- --------
  Training cohort                     41,983
  Temporally held-out test cohort     17,083
  Total                               59,066

The temporal split uses cases through **31 December 2021** for training
and cases from **1 January 2022 to 10 August 2023** for held-out
evaluation.

The underlying clinical data and patient-level processed datasets are
**not redistributed in this repository**. The code therefore expects
users who are authorized to use the source data to provide the required
input files locally. The repository contains selected aggregate research
outputs rather than raw patient-level records or case-level clinical
datasets.

The software license for this repository does not grant rights to the
underlying MOVER dataset. Users are responsible for complying with the
original dataset's access and use conditions.

## Predictive Models

Three final predictive pipelines are evaluated.

### LightGBM

LightGBM uses separate quantile-regression models for P10, P50, and P90
surgical-duration estimation. The final implementation is provided in:

``` text
uncertainty/lightgbm_quantile_uncertainty.py
```

Independent quantile estimates are post-processed to prevent quantile
crossing while leaving P50 unchanged.

### Random Forest

Random Forest is implemented in:

``` text
models/random_forest.ipynb
uncertainty/rf_quantile_uncertainty.py
```

Its P10, P50, and P90 estimates are empirical percentiles of
individual-tree predictions. They therefore represent **between-tree
predictive dispersion**, not native quantile regression or a complete
conditional predictive distribution.

### XGBoost

XGBoost uses separate quantile models for P10, P50, and P90 estimation:

``` text
models/xgboost.ipynb
uncertainty/xgb_quantile_uncertainty.py
```

As with LightGBM, post-processing prevents quantile crossing.

## Predictive Performance

Performance is evaluated exclusively on the **17,083-case temporally
held-out test cohort**.

  -----------------------------------------------------------------------
  Model        P50 MAE (min)   P10 Coverage   P90 Coverage        P10-P90
                                                                 Coverage
  ----------- -------------- -------------- -------------- --------------
  LightGBM         **74.76**         90.19%         89.74%         79.93%

  Random               93.90         60.58%         59.14%         19.72%
  Forest                                                   

  XGBoost              85.13         90.27%         89.83%     **80.09%**

  Nominal                 \-         90.00%         90.00%         80.00%
  target                                                   
  -----------------------------------------------------------------------

LightGBM provides the lowest median-duration prediction error. LightGBM
and XGBoost produce P10-P90 empirical coverage close to the nominal 80%
reference, whereas the Random Forest between-tree approximation
substantially undercovers.

![Predictive interval
coverage](figures/final/02_predictive_interval_coverage.png)

## Risk-Aware MILP Scheduling

For case (i), the duration supplied to the scheduler is

\[ d_i(`\lambda`{=tex})=(1-`\lambda`{=tex})P50_i+`\lambda `{=tex}P90_i,
\]

where (`\lambda`{=tex}`\in[0,1]`{=tex}) controls risk aversion.

The final experiments use:

  Setting                                                  Value
  ----------------------------------------- --------------------
  Surgical cases per experiment                               12
  Operating rooms                                              3
  Nominal room capacity                                  480 min
  Turnover time                                           20 min
  Workload-balance weight (`\beta`{=tex})                   0.10
  Risk-aversion grid                          0.0, 0.1, ..., 1.0
  Duration cap                                           360 min
  Solver time budget                                       300 s
  Target relative MIP gap                                     1%

The MILP jointly determines room assignment, start times, and endogenous
pairwise sequencing. The final implementation is:

``` text
optimization/milp_engine.ipynb
```

Supporting formulation checks are provided in:

``` text
optimization/optimizer_input_verification.ipynb
optimization/sequencing_verification.ipynb
```

## Computational Performance

The three predictive pipelines are evaluated across 11 risk-aversion
settings, producing **33 main MILP instances**.

All 33 instances returned **feasible incumbent schedules** within the
300-second computational budget, but none reached the prespecified 1%
relative MIP-gap target. The median final MIP gap was **56.76%**.

These schedules should therefore be interpreted as **time-limited
feasible incumbents**. Global optimality was not established.

## Realized-Duration Evaluation

Planned model-specific objectives are not treated as direct evidence of
cross-model operational superiority because each predictive model
supplies different duration inputs.

For a common downstream comparison, each returned assignment and
sequence is frozen and replayed using the same **observed surgical
durations**, with the 20-minute turnover retained and **without
hindsight re-optimization**.

The primary evaluation considers (`\lambda `{=tex}`\in `{=tex}{0,0.5,1})
for all three models. Two additional non-overlapping 12-case cohorts are
evaluated at (`\lambda=0.5`{=tex}).

The main realized metrics include:

-   realized overtime;
-   realized makespan;
-   propagated start delay; and
-   number of delayed cases.

At (`\lambda=0.5`{=tex}), Random Forest produces the lowest realized
overtime across all three evaluated cohorts, while the lowest propagated
delay is achieved by XGBoost on the primary cohort and LightGBM on
Cohorts A and B. Superior point-prediction accuracy therefore does not
translate uniformly into dominance across realized scheduling metrics.

On the primary cohort, increasing risk aversion reduces propagated start
delay but increases realized overtime for LightGBM and XGBoost,
illustrating an overtime-delay trade-off.

![Cross-cohort realized overtime-delay
trade-off](figures/final/08_cross_cohort_overtime_delay_tradeoff.png)

## Repository Structure

``` text
uncertainty-aware-or-scheduling/
├── README.md
├── requirements.txt
├── .gitignore
├── preprocessing/
│   ├── pre-processing.py
│   └── feature_engineering.py
├── models/
│   ├── random_forest.ipynb
│   └── xgboost.ipynb
├── uncertainty/
│   ├── lightgbm_quantile_uncertainty.py
│   ├── rf_quantile_uncertainty.py
│   └── xgb_quantile_uncertainty.py
├── optimization/
│   ├── milp_engine.ipynb
│   ├── optimizer_input_verification.ipynb
│   └── sequencing_verification.ipynb
├── evaluation/
│   ├── experiment_runner.ipynb
│   ├── realised_schedule_validation.ipynb
│   ├── realised_model_comparison.ipynb
│   ├── additional_cohort_validation.ipynb
│   ├── duration_cap_audit.ipynb
│   ├── duration_cap_sensitivity.ipynb
│   └── room_metric_semantics_audit.ipynb
├── results/
│   ├── optimization/
│   ├── realized/
│   ├── sensitivity/
│   └── audits/
└── figures/
    └── final/
```

The public repository intentionally excludes legacy Ridge, Monte Carlo,
ablation, rejected development formulations, checkpoint files, raw
clinical data, patient-level processed data, and serialized
preprocessing objects.

## Reproducibility

### Installation

A Python environment is required. Install the project dependencies with:

``` bash
pip install -r requirements.txt
```

A valid **Gurobi installation and license** are required to execute the
MILP experiments.

### Data Preparation

Because the underlying clinical data are not redistributed, reproduce
the data-processing stage only after obtaining authorized access to the
source data.

The preprocessing scripts expect the source files locally and generate
the processed feature datasets required by the predictive pipelines.
Local data paths may need to be configured for the user's environment.

### Suggested Execution Order

The final workflow is organized as follows:

``` text
1. preprocessing/pre-processing.py
2. preprocessing/feature_engineering.py

3. models/random_forest.ipynb
4. models/xgboost.ipynb
5. uncertainty/lightgbm_quantile_uncertainty.py
6. uncertainty/rf_quantile_uncertainty.py
7. uncertainty/xgb_quantile_uncertainty.py

8. optimization/optimizer_input_verification.ipynb
9. optimization/milp_engine.ipynb
10. optimization/sequencing_verification.ipynb

11. evaluation/experiment_runner.ipynb
12. evaluation/realised_schedule_validation.ipynb
13. evaluation/realised_model_comparison.ipynb
14. evaluation/additional_cohort_validation.ipynb
15. evaluation/duration_cap_audit.ipynb
16. evaluation/duration_cap_sensitivity.ipynb
17. evaluation/room_metric_semantics_audit.ipynb
```

Selected frozen aggregate outputs are provided under `results/`, and the
final dissertation figures are provided under `figures/final/`.

## Key Findings

The final experiments support four main conclusions:

1.  **Predictive accuracy and uncertainty calibration are distinct.**
    LightGBM achieves the lowest P50 MAE, while LightGBM and XGBoost
    provide substantially better P10-P90 empirical coverage than the
    Random Forest between-tree approximation.
2.  **Predictive ranking does not determine operational ranking.** The
    model with the lowest point-prediction error does not uniformly
    dominate realized overtime or propagated delay.
3.  **Risk preference changes downstream behavior.** Moving planning
    durations from P50 toward P90 can reduce propagated delay while
    increasing overtime.
4.  **Computational quality matters.** All main MILP runs return
    feasible incumbents, but none establishes global optimality within
    the 300-second budget.

## Limitations

The findings should be interpreted in light of several limitations:

-   retrospective, single-environment data;
-   restricted cohort-level scheduling evaluation;
-   simplified operational constraints relative to a live OR
    environment;
-   Random Forest uncertainty represented by between-tree dispersion
    rather than native quantile regression;
-   sensitivity of high-risk schedules to the duration cap; and
-   time-limited MILP optimization with non-zero final optimality gaps.

Prospective and institution-specific validation would be required before
operational deployment.

## Ethics

The research uses retrospective secondary data. The dissertation's
institutional ethics review determined that formal ethical approval was
waived for this project. No raw patient-level clinical data are
redistributed through this repository.

## Citation

If you use or build on this repository, please cite the associated
dissertation. Bibliographic metadata can be added here once the final
institutional dissertation record is available.

## License

A software license will be specified in the repository `LICENSE` file.
Any such license applies only to original code and repository materials
for which the author holds the relevant rights; it does **not** grant
rights to the underlying MOVER dataset.

## Disclaimer

This repository is an academic research artifact. It is not a clinical
decision-support system and is not intended for direct deployment in
operating-room scheduling without independent validation, governance
review, and institution-specific adaptation.
