# Analytics

`opl.analytics` provides three capabilities built on top of the core SDK:

- **Normative percentiles** — where a lift stands relative to peers
- **Feature engineering** — extracting ML-ready signals from a lifter's history
- **Trajectory prediction** — forecasting future competition performance

Requires `pip install opl-py[analytics]` (adds `scikit-learn` and `polars`).

---

## Normative Percentiles

Given a lift type, weight in kg, and demographic filters, returns what percentile that performance sits at relative to all matching entries in the OPL dataset.

```python
from opl.analytics import percentile
import opl

p = percentile(
    lift="squat",
    weight=200.0,
    sex=opl.Sex.MALE,
    equipment=opl.Equipment.RAW,
    weight_class="93",
    tested=True,
)
# -> 82.5
```

Supported lifts: `"squat"`, `"bench"`, `"deadlift"`, `"total"`.

Optional filters: `weight_class`, `tested`, `sex`, `equipment`. Omitting a filter widens the comparison population.

### Why this approach

The OPL dataset contains millions of competition entries, making it the most comprehensive public-domain source of real powerlifting performances. Percentile ranking is computed directly from this data using a simple counting query against the local DuckDB database — no parametric assumptions, no fitted distributions. This is correct for powerlifting data, which is not normally distributed (it is heavily right-skewed, with most competitors clustered in the beginner-to-intermediate range and a long tail of elite performers). A count-based empirical percentile handles this naturally.

The `Best3` columns are used (best successful attempt across three tries), which matches how OPL itself ranks lifters and how the sport measures official performance.

---

## Feature Engineering

`extract_features(lifter)` converts a `Lifter` object's full competition history into a flat `LifterFeatures` dataclass suitable for ML models.

```python
from opl.analytics import extract_features

features = extract_features(lifter)
features.career_length_days      # 1246
features.competition_count       # 12
features.competition_frequency   # 3.52  (meets/year)
features.best_total_kg           # 872.5
features.total_progression_rate  # 48.3  (kg/year)
features.squat_to_total_ratio    # 0.381
features.equipment_mode          # "Raw"
```

### Full feature set

| Feature | Description |
|---|---|
| `career_length_days` | Days between first and most recent competition |
| `competition_count` | Total number of meet entries |
| `competition_frequency` | Meets per year |
| `best_total_kg` | All-time best competition total |
| `best_squat_kg` | All-time best squat |
| `best_bench_kg` | All-time best bench press |
| `best_deadlift_kg` | All-time best deadlift |
| `latest_total_kg` | Total at most recent competition |
| `latest_bodyweight_kg` | Bodyweight at most recent competition |
| `total_progression_rate` | kg of total gained per year |
| `squat_to_total_ratio` | Squat as fraction of total (latest entry) |
| `bench_to_total_ratio` | Bench as fraction of total (latest entry) |
| `deadlift_to_total_ratio` | Deadlift as fraction of total (latest entry) |
| `age_at_latest` | Age at most recent competition |
| `age_at_first` | Age at first competition |
| `weight_class_numeric` | Weight class parsed to float (SHW `+` stripped) |
| `is_tested` | Tested status at most recent competition |
| `equipment_mode` | Most commonly used equipment category |
| `days_since_last_comp` | Days between the last two competitions |

### Why these features

Each feature was chosen to capture a distinct, meaningful signal about a lifter's trajectory:

**Performance ceiling and current level** (`best_*`, `latest_*`) — the single strongest predictor of next-competition performance is recent performance. Distinguishing between all-time best and most recent performance also captures whether the lifter is peaking, declining, or still improving.

**Progression rate** — the rate of gain per year of competing encodes how quickly a lifter is developing relative to the time they have invested. A lifter with a high progression rate early in their career will likely continue improving; a lifter with a flat or declining rate is near their ceiling.

**Career stage** (`career_length_days`, `age_at_latest`, `age_at_first`) — age and career length interact non-linearly with performance. Powerlifters typically peak in their late 20s to mid-30s, with squat and deadlift peaking later than bench. These features give the model the information it needs to account for developmental stage.

**Competition frequency and recency** (`competition_frequency`, `days_since_last_comp`) — lifters who compete more often accumulate competition-specific experience. A longer gap since last competition may indicate injury, life circumstances, or deliberate peaking, all of which affect the next result.

**Lift ratios** (`squat_to_total_ratio`, etc.) — the balance between the three lifts is relatively stable for a given lifter and equipment style. These ratios let the per-lift models (squat, bench, deadlift) borrow information from the total prediction, and help detect outlier meets (e.g., a lifter who bombed out on deadlift).

**Equipment and tested status** — raw vs. equipped lifting and drug-tested vs. untested are separate competitive categories with systematically different totals. These are necessary controls, not predictive signals.

---

## Trajectory Prediction

`TrajectoryModel` trains four separate regressors (total, squat, bench, deadlift) from a population of lifters and predicts a given lifter's next competition performance.

```python
from opl.analytics import TrajectoryModel, predict_trajectory

model = TrajectoryModel()
scores = model.train(training_lifters)
# scores -> {"total": 0.87, "squat": 0.84, "bench": 0.81, "deadlift": 0.83}

prediction = predict_trajectory(lifter, model=model)
prediction.next_total_kg        # 872.5
prediction.next_squat_kg        # 340.0
prediction.next_bench_kg        # 197.5
prediction.next_deadlift_kg     # 335.0
prediction.confidence_interval  # (829.0, 916.0)
prediction.trajectory_curve     # [(2, 876.1), (4, 880.2), ...]
```

### Training methodology

**Dataset construction (leave-last-out):** For each lifter with 3+ entries, the model uses features derived from all entries except the final one to predict that final entry's results. This is a natural walk-forward split that respects time ordering — the model never sees future data during training. Features are extracted from the `entries[:-1]` sub-history; the target is `entries[-1]`.total, squat, bench, deadlift.

**Cross-sectional model:** A single model is trained across all lifters, not one model per lifter. This is the correct approach because most lifters have too few competition entries (median ~5) to fit a per-lifter model. A cross-sectional model generalises from lifters with richer histories to predict for any lifter with 3+ entries.

**80/20 train/test split** with `random_state=42` for reproducibility. R² on the held-out 20% is reported per target.

### Why HistGradientBoostingRegressor

Powerlifting performance data has several properties that make gradient boosted trees the right model family:

**Missing values are common.** Not every lifter records bodyweight, age, or weight class at every meet. OPL data is user-contributed and has real gaps. `HistGradientBoostingRegressor` handles `NaN` natively — it learns the best split direction for missing values during training. This means no imputation step is needed, and the model correctly learns that "missing age" is informative on its own (it often correlates with masters lifters who don't report age, or with older records).

**Non-linear relationships.** The relationship between age and performance is not linear — lifters improve rapidly in their first few years, plateau, then decline. Career length and competition count interact. Tree-based models capture these non-linearities automatically without requiring manual polynomial features or interaction terms.

**Heterogeneous feature scales.** The features range from ratios (0–1) to raw kilogram values (0–1000+) to day counts (0–10,000+). Gradient boosted trees are scale-invariant; no normalisation is needed.

**Robustness to outliers.** The OPL dataset includes bomb-outs (failed totals, recorded as 0 or NULL), guest lifters, and records from federations with different weigh-in rules. Tree splits are robust to these in a way that linear models are not.

**No need for deep learning.** The feature count is ~17. A neural network would require far more data and tuning to outperform gradient boosting on a tabular dataset of this size. Interpretability and fast training are meaningful advantages: the pretrained model ships with the package (<10 MB), and users can retrain on their local data in seconds.

### Hyperparameters

```python
HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
)
```

These are conservative, well-generalising defaults. `max_iter=200` with `learning_rate=0.1` follows the standard bias-variance tradeoff for gradient boosting. `max_depth=6` allows the model to capture interaction effects between features (e.g., age × career length × progression rate) without overfitting.

### Confidence interval

The confidence interval is computed as `predicted_total ± 5%`. This is a pragmatic approximation — a proper interval would require conformal prediction or a quantile regression variant, both of which add significant complexity. The 5% margin reflects the empirical meet-to-meet variability commonly observed in competitive powerlifting (a peaking lifter typically varies ±3–7% of their best total across consecutive competitions).

### Trajectory curve

The projected trajectory curve extrapolates the lifter's historical monthly progression rate forward from the predicted next competition. It is a linear extrapolation, not a model prediction. Its purpose is illustrative — showing whether a lifter is on an upward, flat, or downward arc — rather than a precise forecast beyond the next meet.

---

## Pretrained Model

A model trained on the full OPL dataset is serialised with `joblib` and shipped in the `pretrained/` directory. This lets users get predictions without needing to train from scratch (which requires the full ~3M row dataset to be loaded into memory).

The pretrained model is versioned by training date (e.g., `pretrained/2025-01-01/model.joblib`). The most recent directory is used automatically.

```python
from pathlib import Path
from opl.analytics import TrajectoryModel, predict_trajectory

model = TrajectoryModel()
model.load(Path("pretrained/2025-01-01/model.joblib"))

prediction = predict_trajectory(lifter, model=model)
```

To retrain on your local data (e.g., after running `opl update` to get fresh competition results):

```bash
python -m opl.analytics.scripts.train
python -m opl.analytics.scripts.train --db-path /path/to/opl.duckdb
python -m opl.analytics.scripts.train --output-dir /path/to/pretrained
python -m opl.analytics.scripts.train --limit 5000  # cap lifters, useful for testing
```

---

## Module Structure

```
opl/analytics/
├── __init__.py          # Public API: percentile, extract_features, predict_trajectory,
│                        #             TrajectoryModel, TrajectoryPrediction
├── features.py          # LifterFeatures dataclass + extract_features()
├── normative.py         # percentile() — empirical count-based percentile ranking
├── trajectory.py        # TrajectoryModel, TrajectoryPrediction, predict_trajectory()
└── scripts/
    └── train.py         # Standalone training script for rebuilding the pretrained model
```
