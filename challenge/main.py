# %% [markdown]
# # Mirico Technical Interview – Methane Threshold Baseline
# 
# This notebook follows the assignment instructions:
# 1. Exploratory Data Analysis (EDA)
# 2. Flat-threshold anomaly detection on CH₄ (ppm)
# 3. Time-based performance evaluation vs truth windows
# 4. Forward-looking improvements
# 
# ## Key constraints from the brief
# - Use only `measurement_validity == 10` (good data)
# - Evaluate as a single flat time series (do not evaluate per retro)
# - Detection must use a single global threshold (one value)
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", 50)

DATA_MEAS = Path("measurement_data.csv")
DATA_TRUTH = Path("truth_data.csv")
SITE_IMG = Path("Mirico_site_layout.png")

# 0) Load data
meas_raw = pd.read_csv(DATA_MEAS)
truth_raw = pd.read_csv(DATA_TRUTH)

print("measurement_data.csv:", meas_raw.shape)
print("truth_data.csv:", truth_raw.shape)

meas_raw.head()


# %% [markdown]
# ## Site layout (context only)
# The NG5 sensor and retros (R1–R14) are shown below (provided by Mirico).

# %%
from PIL import Image
from IPython.display import display

if SITE_IMG.exists():
    display(Image.open(SITE_IMG))
else:
    print("Site layout image not found:", SITE_IMG)


# %% [markdown]
# ## 1) Cleaning + feature derivations
# Wind speed/direction are derived for EDA only (not used for thresholding).

# %%
meas = meas_raw.copy()

# Parse timestamps
meas["timestamp"] = pd.to_datetime(meas["timestamp"], errors="coerce")
meas = meas.dropna(subset=["timestamp", "ch4_ppm"]).sort_values("timestamp")

# Keep only good data
meas = meas[meas["measurement_validity"] == 10].copy()

# Derive horizontal wind speed + direction (EDA only)
wx = meas["windx_m_per_s"].astype(float)
wy = meas["windy_m_per_s"].astype(float)
meas["wind_horiz_speed_m_per_s"] = np.hypot(wx, wy)
meas["wind_horiz_dir_deg"] = (np.degrees(np.arctan2(wy, wx)) + 360.0) % 360.0

meas.shape, meas.head()


# %% [markdown]
# ## Parse truth windows
# Use the provided start / end columns (already fully qualified timestamps).

# %%
truth = truth_raw.copy()
truth["start"] = pd.to_datetime(truth["start"], errors="coerce")
truth["end"] = pd.to_datetime(truth["end"], errors="coerce")
truth = truth.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)

truth[["start", "end", "kg/h"]].head()


# %% [markdown]
# ## 2) Exploratory Data Analysis (EDA)
# A typical global background methane concentration is ~2 ppm. We look for excursions above this background.

# %%
meas[["ch4_ppm", "temperature_k", "pressure_torr", "wind_horiz_speed_m_per_s"]].describe()

# %%
# Missingness & retro counts (retro is contextual only; evaluation is flat)
display(meas.isna().mean().sort_values(ascending=False).head(12))
display(meas["retro_name_id"].value_counts())


# %%
# CH4 over time (10s max to reduce overplotting)
ch4_10s = meas.set_index("timestamp")["ch4_ppm"].resample("10s").max()
plt.figure()
ch4_10s.plot()
plt.xlabel("Time")
plt.ylabel("CH4 (ppm)")
plt.title("CH4 concentration over time (10s max)")
plt.tight_layout()
plt.show()


# %%
# CH4 distribution (clip extreme tail for readability)
clip_val = meas["ch4_ppm"].quantile(0.999)
plt.figure()
plt.hist(meas["ch4_ppm"].clip(upper=clip_val), bins=80)
plt.xlabel("CH4 (ppm) (clipped at 99.9%)")
plt.ylabel("Count")
plt.title("Distribution of CH4 concentration")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Notes from EDA
# - **Normal:** values cluster tightly around ~2 ppm with modest variability.
# - **Unusual:** sharp spikes and sustained elevated periods, sometimes reaching tens of ppm and occasionally > 100 ppm.
# - **Data limitations:** sensor cycles between retro paths; the raw data are not a continuous 1 Hz time series.
# - Because the brief asks for a flat timeseries evaluation, we avoid per-retro modelling here.
# 

# %% [markdown]
# ## 3) Label data with truth windows + make 1 Hz bins
# For pragmatic time-based evaluation, we aggregate to per-second maxima of CH₄ across all samples occurring within that second.
# 

# %%
def add_truth_label(meas_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a truth label (0/1) to each measurement row if its timestamp lies within any truth window."""
    starts = truth_df["start"].values.astype("datetime64[ns]")
    ends = truth_df["end"].values.astype("datetime64[ns]")
    ts = meas_df["timestamp"].values.astype("datetime64[ns]")

    # Find the last window start before each timestamp
    idx = np.searchsorted(starts, ts, side="right") - 1
    in_window = (idx >= 0) & (ts <= ends[idx])

    out = meas_df.copy()
    out["truth"] = in_window.astype(int)
    return out


def to_second_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to 1 Hz (per-second) bins using max CH4 and max truth label."""
    tmp = df[["timestamp", "ch4_ppm", "truth"]].copy()
    tmp["t_sec"] = tmp["timestamp"].dt.floor("s")
    sec = (
        tmp.groupby("t_sec")
        .agg(ch4_ppm=("ch4_ppm", "max"), truth=("truth", "max"), n=("ch4_ppm", "size"))
        .reset_index()
        .rename(columns={"t_sec": "timestamp"})
        .sort_values("timestamp")
    )
    return sec


meas_l = add_truth_label(meas, truth)
sec = to_second_bins(meas_l)

sec.head(), sec.shape


# %% [markdown]
# Coverage note: the sensor is not sampling every second. We evaluate only on observed seconds to avoid inventing values in the gaps.

# %%
full_seconds = int((sec["timestamp"].max() - sec["timestamp"].min()).total_seconds()) + 1
coverage = sec.shape[0] / full_seconds
print(f"Observed seconds: {sec.shape[0]:,} / {full_seconds:,} (~{coverage:.1%} coverage)")


# %% [markdown]
# ## 4) Flat-threshold detector + tuning
# We sweep candidate thresholds and compute simple time-bin metrics:
# - **True positive time** = fraction of truth-anomalous seconds that are detected (recall)
# - **False positive time** = fraction of detected-anomalous seconds that do not overlap truth (1 - precision)
# 

# %%
def compute_metrics(sec_df: pd.DataFrame, threshold_ppm: float) -> dict:
    pred = (sec_df["ch4_ppm"] > threshold_ppm).astype(int)
    truth_lbl = sec_df["truth"].astype(int)

    tp = int(((pred == 1) & (truth_lbl == 1)).sum())
    fp = int(((pred == 1) & (truth_lbl == 0)).sum())
    fn = int(((pred == 0) & (truth_lbl == 1)).sum())
    tn = int(((pred == 0) & (truth_lbl == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return dict(
        threshold_ppm=float(threshold_ppm),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positive_time=float(recall),
        false_positive_time=float(fp / (tp + fp) if (tp + fp) else 0.0),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def tune_threshold(sec_df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = [compute_metrics(sec_df, float(t)) for t in thresholds]
    return pd.DataFrame(rows)


threshold_grid = np.round(np.arange(1.9, 6.01, 0.01), 2)
tuned = tune_threshold(sec, threshold_grid)

# Option A: raw "best F1" (often gives unacceptably high false positives)
best_f1 = tuned.sort_values(["f1", "recall"], ascending=False).iloc[0]

# Option B: choose best F1 subject to a minimum precision (more practical)
min_precision = 0.70
best_practical = (
    tuned[tuned["precision"] >= min_precision]
    .sort_values(["f1", "recall"], ascending=False)
    .iloc[0]
)

display(best_f1)
display(best_practical)


# %%
plt.figure()
plt.plot(tuned["threshold_ppm"], tuned["precision"], label="precision")
plt.plot(tuned["threshold_ppm"], tuned["recall"], label="recall")
plt.plot(tuned["threshold_ppm"], tuned["f1"], label="f1")
plt.xlabel("Threshold (ppm)")
plt.ylabel("Metric")
plt.title("Threshold sweep (1 Hz time bins)")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Threshold choice used for deliverables
# A pure F1 optimum can occur at a very low threshold, which detects most truth time but produces excessive false positives.
# For a more usable baseline, select the best threshold with precision ≥ 0.70.
# This is still a single global threshold; the constraint is only used to pick a point on the precision–recall trade-off curve.
# 

# %%
threshold_ppm = float(best_practical["threshold_ppm"])
threshold_ppm


# %% [markdown]
# ## 5) Convert exceedances to detected anomaly windows
# We convert per-second threshold exceedances into windows by merging exceedances separated by short gaps (`gap_tolerance_s`)
# and dropping extremely short events (`min_duration_s`). This is post-processing only; the detector remains a single flat threshold.
# 

# %%
def exceedances_to_windows(
    sec_df: pd.DataFrame,
    threshold_ppm: float,
    gap_tolerance_s: int = 30,
    min_duration_s: int = 10,
) -> pd.DataFrame:
    df = sec_df[["timestamp", "ch4_ppm"]].copy()
    df["anomaly"] = (df["ch4_ppm"] > threshold_ppm).astype(int)

    pos = df[df["anomaly"] == 1].copy()
    if pos.empty:
        return pd.DataFrame(columns=["start", "end", "duration_s"])

    pos = pos.sort_values("timestamp")
    dt = pos["timestamp"].diff().dt.total_seconds().fillna(0)
    group_id = (dt > (gap_tolerance_s + 1)).cumsum()

    windows = (
        pos.groupby(group_id)
        .agg(start=("timestamp", "min"), end=("timestamp", "max"))
        .reset_index(drop=True)
    )
    windows["duration_s"] = (windows["end"] - windows["start"]).dt.total_seconds().astype(int) + 1
    windows = windows[windows["duration_s"] >= min_duration_s].reset_index(drop=True)
    return windows


detected_windows = exceedances_to_windows(sec, threshold_ppm, gap_tolerance_s=30, min_duration_s=10)
detected_windows.head(), detected_windows.shape


# %% [markdown]
# ### Visual alignment with truth
# 
# Plot a shorter window for readability and show truth release windows as thick bars.

# %%

start_plot = sec["timestamp"].min()
end_plot = start_plot + pd.Timedelta(hours=12)
view = sec[(sec["timestamp"] >= start_plot) & (sec["timestamp"] <= end_plot)].copy()

plt.figure()
plt.plot(view["timestamp"], view["ch4_ppm"])
plt.axhline(threshold_ppm, linestyle="--", label=f"threshold={threshold_ppm:.2f} ppm")
plt.xlabel("Time")
plt.ylabel("CH4 (ppm)")
plt.title("CH4 (1 Hz max) with threshold – first 12 hours")

# Truth shading (as thick horizontal segments at the threshold level)
for _, row in truth.iterrows():
    if row["end"] < start_plot or row["start"] > end_plot:
        continue
    plt.plot([row["start"], row["end"]], [threshold_ppm, threshold_ppm], linewidth=6)

plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6) Performance evaluation (time-based)
# We report time-based metrics on the per-second series:

# %%
metrics = compute_metrics(sec, threshold_ppm)
metrics


# %% [markdown]
# ## 7) Export deliverables
# - `labelled_seconds.csv`: 1 Hz dataframe with truth + predicted anomaly  
# - `detected_windows.csv`: merged anomaly windows  
# - `threshold_sweep_metrics.csv`: sweep results across thresholds
# 

# %%
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

labelled = sec.copy()
labelled["pred_anomaly"] = (labelled["ch4_ppm"] > threshold_ppm).astype(int)

labelled_path = OUT_DIR / "labelled_seconds.csv"
windows_path = OUT_DIR / "detected_windows.csv"
sweep_path = OUT_DIR / "threshold_sweep_metrics.csv"  # export sweep metrics

labelled.to_csv(labelled_path, index=False)
detected_windows.to_csv(windows_path, index=False)
tuned.to_csv(sweep_path, index=False)  

labelled_path, windows_path, sweep_path


# %% [markdown]
# ## 8) Forward-looking improvements (high-level)
# If given more time, I would improve this baseline by:
# - Sampling / alignment improvements robust to irregular sampling
# - Per-retro normalisation
# - Adaptive thresholds
# - Use wind features
# - Better windowing (hysteresis)
# - Event-based evaluation (IoU, latency, stratification by kg/h)
# 


