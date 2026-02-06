# Mirico methane threshold baseline

## What I did (approach)
1. **Data cleaning**
   - Parsed timestamps and dropped rows with missing timestamp/CH4.
   - Kept only `measurement_validity == 10` (good quality), per the brief.
   - Derived horizontal wind speed and direction **for EDA only** (not used in the threshold detector).

2. **EDA**
   - Background CH₄ is ~2 ppm. The dataset is tightly clustered around this background with occasional spikes reaching tens of ppm (rarely >100 ppm).
   - Measurements cycle through different retro paths, so the dataset is not a continuous 1 Hz series. I therefore evaluated only on *observed seconds*.

3. **Truth labelling + 1 Hz binning**
   - Labeled each measurement as truth-anomalous if its timestamp falls within any provided truth release window.
   - Aggregated to per-second bins using **max CH₄** and **max truth label** within each second. This gives a pragmatic time series for the requested time-based scoring.

4. **Thresholding + tuning**
   - Applied a single global threshold on the 1 Hz CH₄ series: `pred = (CH4 > threshold)`.
   - Swept thresholds from 1.9–6.0 ppm (0.01 ppm step) and computed:
     - **True positive time** = recall = TP/(TP+FN)
     - **False positive time** = FP/(TP+FP) = 1 - precision
   - A pure F1 optimum can occur at very low thresholds, which achieves high recall but produces excessive false positives.
   - For a more usable baseline, I selected the **best F1 subject to precision ≥ 0.70**, giving:
     - **threshold = 2.23 ppm**
     - precision ≈ 0.709, recall ≈ 0.257, F1 ≈ 0.377

5. **Windows**
   - Converted exceedance seconds into anomaly windows by merging exceedances separated by ≤30 s gaps and dropping events shorter than 10 s. This is post-processing only; the detector is still a single threshold.

## Key assumptions / limitations
- The evaluation is *flat* across all retros (as requested). In reality, retros likely have different baselines/noise and different plume intersection probability.
- I evaluated only seconds with observed data (sensor cycle causes many missing seconds). Filling gaps would require additional assumptions about sensor behaviour.

## Deliverables
- `main.ipynb` – notebook with EDA, tuning, plots, and discussion
- `main.py` – script version (reproducible runs + CSV outputs)
- `outputs/labelled_seconds.csv` – 1 Hz labels (`truth`, `pred_anomaly`)
- `outputs/detected_windows.csv` – merged anomaly windows
- `outputs/threshold_sweep_metrics.csv` – sweep table for transparency

## If I had more time (next steps)
- Per-retro baseline normalisation and/or adaptive thresholds (rolling quantiles/MAD).
- Use wind direction/speed as gating or features to reduce false positives.
- Event-based evaluation (overlap/IoU), detection latency, and stratification by release rate (kg/h).
