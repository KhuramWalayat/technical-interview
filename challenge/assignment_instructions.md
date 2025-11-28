# Assignment Instructions

## Overview

You are provided with methane concentration measurements and environmental data from an industrial monitoring system, along with known time windows of controlled releases.

The aim of this exercise is to:

- Explore and understand the data  
- Implement a simple, flat threshold based anomaly detection approach  
- Use the truth data to tune your method  
- Reflect on how you would improve the approach with more time  

This assignment is designed to be achievable within the allotted time and to form the basis of technical discussion in the next interview stage.

---

## Time Expectations

Please spend between **2–4 hours** on this task.  
We encourage you to work pragmatically within this limit rather than aiming for perfection.

Part of the assessment is how you prioritise and allocate your time. If you choose not to pursue certain improvements due to time constraints, please note this and briefly explain your reasoning.

We are not looking for a perfect solution. Instead, we are interested in:

- Your reasoning process  
- How you break down the problem  
- Your coding clarity  
- How you think about model behaviour and performance  

---

## Data Files

Two files are provided for this assignment:

### measurement_data.csv  
Contains observed methane concentration data and relevant environmental readings (units are expressed in the column headings).

**Notes:**
- `windx`, `windy`, `windz` represent wind vector components measuring on-site wind movement as recorded by an anemometer  
- These components should be used to derive (just to show understanding - not used in the thresholding exercise):
  - **Horizontal wind speed**
  - **Horizontal Wind direction**
- `measurement_validity` indicates data quality. A value of `10` means good data; assume anything else is not of sufficient quality  
- 'retro_name_id' is the name of the retro reflector (mirror) used by the sensor to reflect the laser
- A typical global background concentration of methane is roughly **2 parts per million (ppm)**  

### truth_data.csv  
Contains time windows of actual controlled methane releases representing anomalous emission events.
- 'kg/h' refers to the rate of the controlled release

### Mirico_site_layout.png
An image of the actual site layout of Mirico retro reflectors (R1 to R14), and the sensor itself (NG5) for context.

---

## Required Tasks

### 1. Exploratory Data Analysis (EDA)

Perform exploratory analysis to understand the data, including:

- Basic summary statistics and distributions  
- Visualisation of methane concentration over time  
- Inspection of data quality and missing values  
- Any notable patterns or anomalies

Briefly describe:

- What “normal” vs “unusual” behaviour looks like  
- Any challenges or limitations you observe in the data  

---

### 2. Simple Anomaly Detection Using Thresholding

Implement a straightforward anomaly detection approach using **flat thresholding** on methane concentration values.

You should:

- Define a single threshold that flag elevated readings  
- Convert threshold exceedances into time windows of anomalous behaviour  
- Use the truth data to tune and justify your chosen threshold

The objective is not complexity, but clarity, transparency, and reasoning.

---

### 3. Performance Evaluation

Compare your detected anomaly windows against the provided truth windows.

To reduce time spent devising evaluation schemes, you may use a simple time-based comparison between your detected labels and the truth windows.

For example, you might report:

- The percentage of true anomalous time correctly detected (true positive time)  
- The percentage of detected anomalous time that does not correspond to a true anomaly (false positive time)
- Evaluate the data as a flat timeseries. For this work, do not break down by retro (reflective mirror)

This is intended as a pragmatic baseline. You are welcome to supplement this with additional metrics or visual comparisons if you wish.

Briefly explain:

- How well your approach aligns with the truth data  
- Where and why mismatches occur  

---

### 4. Forward-Looking Design (High-Level Only)

Provide a short high-level description of how you would improve the anomaly detection approach if given more time.

This may include:

- Additional preprocessing steps  
- Feature engineering ideas  
- More advanced modelling strategies  
- Ways to incorporate environmental data (e.g. wind)

No implementation is required for this section — we are interested in your conceptual roadmap and technical thinking.

---

## Deliverables

Please submit:

### 1. A Jupyter Notebook or Python script containing:
- Exploratory Data Analysis (EDA)  
- Your detection logic  
- Evaluation and discussion  (This will include a final threshold value with associated performance statistics)

### 2. A csv output of labelled data

### 3. A short README including (max ~500 words):
- Your approach and key assumptions  
- How you selected and tuned your threshold  
- What worked well and what didn’t  
- Your proposed next steps  
- Where relevant, briefly comment on how you prioritised tasks within the time available

---

## Key Principle

This assignment is intentionally scoped to balance technical effort with time constraints. The simplicity of the detection method is deliberate — what matters most is your reasoning and your ability to evaluate and improve it.

Your work will form the foundation of a deeper technical discussion in the next interview stage, where we will explore your design decisions and potential alternative approaches.
