# Analytics

`opl.analytics` provides three capabilities built on top of the core SDK:

- **Normative percentiles** — where a lift stands relative to peers
- **Feature engineering** — extracting ML-ready signals from a lifter's history
- **Trajectory prediction** — forecasting future competition performance across multiple model approaches

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
features.recent_avg_total_kg     # 861.2 (average of last 3 meets)
features.recent_progression_rate # 35.0  (kg/year over last 2 years)
features.total_std_kg            # 12.4  (standard deviation of totals)
features.meets_since_peak        # 2     (meets since all-time best)
features.squat_to_total_ratio    # 0.381
features.equipment_mode          # "Raw"
```

### Full feature set

| Feature                   | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| `career_length_days`      | Days between first and most recent competition             |
| `competition_count`       | Total number of meet entries                               |
| `competition_frequency`   | Meets per year                                             |
| `best_total_kg`           | All-time best competition total                            |
| `best_squat_kg`           | All-time best squat                                        |
| `best_bench_kg`           | All-time best bench press                                  |
| `best_deadlift_kg`        | All-time best deadlift                                     |
| `latest_total_kg`         | Total at most recent competition                           |
| `latest_bodyweight_kg`    | Bodyweight at most recent competition                      |
| `total_progression_rate`  | kg of total gained per year (over full career)             |
| `squat_to_total_ratio`    | Squat as fraction of total (latest entry)                  |
| `bench_to_total_ratio`    | Bench as fraction of total (latest entry)                  |
| `deadlift_to_total_ratio` | Deadlift as fraction of total (latest entry)               |
| `age_at_latest`           | Age at most recent competition                             |
| `age_at_first`            | Age at first competition                                   |
| `weight_class_numeric`    | Weight class parsed to float (SHW `+` stripped)            |
| `is_tested`               | Tested status at most recent competition                   |
| `equipment_mode`          | Most commonly used equipment category                      |
| `days_since_last_comp`    | Days between the last two competitions                     |
| `recent_avg_total_kg`     | Average total across the last 3 meets                      |
| `recent_progression_rate` | kg/year gained over the last ~2 years                      |
| `total_std_kg`            | Standard deviation of all competition totals (consistency) |
| `meets_since_peak`        | Number of meets since the lifter's all-time best total     |

### Why these features

Each feature was chosen to capture a distinct, meaningful signal about a lifter's trajectory:

**Performance ceiling and current level** (`best_*`, `latest_*`) — the single strongest predictor of next-competition performance is recent performance. Distinguishing between all-time best and most recent performance also captures whether the lifter is peaking, declining, or still improving.

**Progression rate** — `total_progression_rate` (full career) and `recent_progression_rate` (last 2 years) together capture both the long-term development arc and whether momentum has changed recently. A lifter who was gaining rapidly but has plateaued is a meaningfully different case from one who is still accelerating.

**Recency signals** (`recent_avg_total_kg`, `meets_since_peak`) — averaging the last 3 meets smooths over single-meet variance (a bad day, a conservative attempt selection). `meets_since_peak` detects whether the lifter is actively chasing their best or has been in a post-peak period.

**Consistency** (`total_std_kg`) — a high standard deviation relative to total indicates an inconsistent competitor, which is a distinct risk factor for prediction. A consistent lifter with low variance is much more predictable than one who ranges ±50 kg between meets.

**Career stage** (`career_length_days`, `age_at_latest`, `age_at_first`) — age and career length interact non-linearly with performance. Powerlifters typically peak in their late 20s to mid-30s, with squat and deadlift peaking later than bench. These features give the model the information it needs to account for developmental stage.

**Competition frequency and recency** (`competition_frequency`, `days_since_last_comp`) — lifters who compete more often accumulate competition-specific experience. A longer gap since last competition may indicate injury, life circumstances, or deliberate peaking, all of which affect the next result.

**Lift ratios** (`squat_to_total_ratio`, etc.) — the balance between the three lifts is relatively stable for a given lifter and equipment style. These ratios let the per-lift models (squat, bench, deadlift) borrow information from the total prediction, and help detect outlier meets (e.g., a lifter who bombed out on deadlift).

**Equipment and tested status** — raw vs. equipped lifting and drug-tested vs. untested are separate competitive categories with systematically different totals. These are necessary controls, not predictive signals.

---

## Trajectory Prediction

The trajectory module provides multiple ML approaches for predicting a lifter's next competition performance. All approaches share the same training data and interface; the registry lets you select, compare, and swap between them.

### Quick start

```python
from opl.analytics import TrajectoryModel, predict_trajectory

# TrajectoryModel is an alias for GradientBoostingModel (the default)
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

### Using the registry

```python
from opl.analytics import get_approach, get_all_approaches, list_approaches

# List all available approaches
list_approaches()
# -> ["gradient_boosting", "quantile_gbt"]

# Get and use a specific approach
cls = get_approach("quantile_gbt")
model = cls()
model.train(training_lifters)
prediction = model.predict(lifter)

# Train all approaches and compare
for name, cls in get_all_approaches().items():
    model = cls()
    scores = model.train(training_lifters)
    print(f"{name}: total R²={scores['total']:.4f}")
```

### Predicting with a target date or bodyweight

Any approach accepts optional `target_date` and `target_bodyweight_kg` arguments:

```python
prediction = model.predict(
    lifter,
    target_date=date(2026, 6, 1),
    target_bodyweight_kg=93.0,
)
prediction.target_date           # date(2026, 6, 1)
prediction.target_bodyweight_kg  # 93.0
```

If omitted, `target_date` defaults to the lifter's average competition interval from their last meet, and `target_bodyweight_kg` defaults to their most recent recorded bodyweight.

---

## Approaches

### Gradient Boosted Trees (`gradient_boosting`)

The default approach. Uses `HistGradientBoostingRegressor` from scikit-learn. Four separate models are trained (total, squat, bench, deadlift). Confidence intervals are computed as `predicted ± 1.96 × residual_std` (falling back to ±5% when residual std is zero). Trajectory curve uses model-based projection at 6 future time points.

```python
from opl.analytics.trajectory.gradient_boosting import GradientBoostingModel
```

**Why HistGradientBoostingRegressor:** Missing values are common in OPL data — not every lifter records bodyweight, age, or weight class at every meet. `HistGradientBoostingRegressor` handles `NaN` natively; it learns the best split direction for missing values during training, so no imputation step is needed. Relationships between age, career length, and performance are non-linear; tree-based models capture these automatically. Features span vastly different scales (ratios 0–1 vs. kg values 0–1000+) but gradient boosted trees are scale-invariant. The OPL dataset includes bomb-outs and federation anomalies; tree splits are robust to outliers.

**Hyperparameters:**

```python
HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42)
```

---

### Quantile GBT (`quantile_gbt`)

Extends the gradient boosting approach by training additional quantile regression models at the 10th and 90th percentiles for the total. These provide **per-prediction confidence intervals** that naturally widen for uncertain cases (sparse history, unusual profiles) and narrow for predictable ones — unlike the global residual-std approach used by the other models.

```python
from opl.analytics.trajectory.quantile_gbt import QuantileGBTModel
```

**When to prefer this:** When calibrated uncertainty estimates matter — e.g., showing a user that their next total could range anywhere from 500–650 kg vs. a tighter 580–620 kg window based on their specific history.

---

## Training methodology

### Dataset construction

For each lifter with `min_entries` or more competition entries, the training pipeline generates one sample per consecutive entry pair. For a lifter with 5 entries `[e0, e1, e2, e3, e4]`, this yields 4 training samples:

| Features extracted from | Target |
| ----------------------- | ------ |
| `[e0]`                  | `e1`   |
| `[e0, e1]`              | `e2`   |
| `[e0, e1, e2]`          | `e3`   |
| `[e0, e1, e2, e3]`      | `e4`   |

This walk-forward scheme respects time ordering — the model never sees future data during training — and maximises training samples from a dataset where most lifters have few entries.

### Feature row

Each training row is a 23-element vector: 21 history features (from `_FEATURE_KEYS`) plus two context features appended at prediction time — `days_to_target` (days until the competition being predicted) and `target_bodyweight_kg` (the bodyweight the lifter is targeting).

### Train/test split

80/20 split with `random_state=42` for reproducibility. R² on the held-out 20% is reported per target (total, squat, bench, deadlift).

### Cross-sectional model

A single model is trained across all lifters, not one model per lifter. This is the correct approach because most lifters have too few competition entries (median ~5) to fit a per-lifter model. A cross-sectional model generalises from lifters with richer histories to predict for any lifter with 3+ entries.

---

## Confidence interval

| Approach            | Method                                                 |
| ------------------- | ------------------------------------------------------ |
| `gradient_boosting` | `predicted ± 1.96 × residual_std` on held-out test set |
| `quantile_gbt`      | 10th and 90th percentile quantile regression models    |

The 5% fallback (`predicted × 0.05`) is used only when residual std is zero (e.g., trivially small training sets). The 5% margin reflects the empirical meet-to-meet variability commonly observed in competitive powerlifting (a peaking lifter typically varies ±3–7% of their best total across consecutive competitions).

---

## Trajectory curve

All approaches project a 6-point trajectory curve from the predicted next competition outward over 12 months. Each point is `(month_offset: int, predicted_total: float)`. The curve is generated by running the trained model at 6 evenly spaced future time points (`[2, 4, 6, 8, 10, 12]` months), holding bodyweight fixed. This gives a model-driven view of the lifter's trajectory rather than a simple linear extrapolation.

`project_trajectory_linear()` is also available as a lightweight alternative that extrapolates historical progression rate forward without calling the model.

---

## Pretrained Model

A model trained on the full OPL dataset is serialised and shipped in the `pretrained/` directory. The pretrained model is versioned by training date and approach name:

```
pretrained/
├── gradient_boosting/
│   └── 2026-03-14/
│       └── model.joblib
└── quantile_gbt/
    └── 2026-03-14/
        └── model.joblib
```

This lets users get predictions without needing to train from scratch (which requires the full ~3M row dataset to be loaded into memory).

```python
from pathlib import Path
from opl.analytics import TrajectoryModel, predict_trajectory

model = TrajectoryModel()
model.load(Path("pretrained/gradient_boosting/2026-03-14/model.joblib"))

prediction = predict_trajectory(lifter, model=model)
```

To retrain on your local data (e.g., after running `opl update` to get fresh competition results):

```bash
python -m opl.analytics.scripts.train
python -m opl.analytics.scripts.train --approach gradient_boosting
python -m opl.analytics.scripts.train --db-path /path/to/opl.duckdb
python -m opl.analytics.scripts.train --output-dir /path/to/pretrained
python -m opl.analytics.scripts.train --min-meets 4
```

---

## Module Structure

```
opl/analytics/
├── __init__.py          # Public API: percentile, extract_features, predict_trajectory,
│                        #   TrajectoryModel, TrajectoryPrediction, get_approach,
│                        #   list_approaches, get_all_approaches
├── features.py          # LifterFeatures dataclass + extract_features()
├── normative.py         # percentile() — empirical count-based percentile ranking
├── trajectory/          # Multi-approach trajectory prediction package
│   ├── __init__.py      # Package API + predict_trajectory() convenience function
│   ├── base.py          # BaseTrajectoryModel ABC, TrajectoryPrediction dataclass,
│   │                    #   shared training/feature utilities
│   ├── registry.py      # @register decorator + get_approach / list_approaches
│   ├── gradient_boosting.py  # GradientBoostingModel (default)
│   └── quantile_gbt.py       # QuantileGBTModel
└── scripts/
    └── train.py         # Standalone training script for rebuilding pretrained models
```
