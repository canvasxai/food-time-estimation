# Meal-to-CGM Event Matching Algorithm

## Overview

This algorithm matches user-logged meals to CGM-detected glucose events using a multi-factor scoring system that considers temporal proximity, physiological plausibility, and meal composition.

---

## The Mathematical Framework

### 1. Score Components

The composite matching score is computed as:

```
S_composite = w₁·S_time + w₂·S_physio + w₃·S_size
```

Where:
- `w₁ = 0.5` (time proximity weight)
- `w₂ = 0.3` (physiological plausibility weight)
- `w₃ = 0.2` (meal size match weight)

---

### 2. Time Proximity Score (S_time)

Measures how close the logged meal time is to the detected event time.

```
Δt = t_meal - t_detected  (in minutes)
```

**Scoring function:**
```
If -30 ≤ Δt ≤ 90:
    S_time = 1 - ((|Δt - 30|) / 120)²
Else:
    S_time = max(0, 1 - (|Δt| / 180)²)
```

**Rationale:**
- Meals are often logged 0-60 minutes AFTER eating begins
- Detection algorithms may identify the glucose rise slightly BEFORE or AFTER the meal is logged
- The "sweet spot" is when `Δt ≈ 30 min` (meal logged ~30 min after detection starts)

---

### 3. Physiological Plausibility Score (S_physio)

Checks if the time-to-peak matches what we'd expect given meal composition.

**Expected Time to Peak:**
```
TTP_expected = 30 + 3·fiber + 1.5·protein + 2·fat  (minutes)
```
Clamped to [30, 180] minutes.

**Actual Time to Peak:**
```
TTP_actual = t_peak - t_meal
```

**Scoring function:**
```
error = |TTP_actual - TTP_expected|
S_physio = max(0, 1 - (error / 120)²)
```

**Rationale:**
- Pure glucose peaks in ~30 min
- Fiber, protein, and fat slow gastric emptying
- A meal with 15g protein, 10g fat, 5g fiber → expect peak in ~80 min

---

### 4. Meal Size Match Score (S_size)

Uses Glycemic Load Proxy to match meal magnitude to event type.

**Glycemic Load Proxy:**
```
net_carbs = max(0, carbs - 0.5·fiber)
dampening = 1 / (1 + 0.02·protein + 0.01·fat)
GL_proxy = net_carbs × dampening
```

**Scoring based on event type:**

| Event Type | GL_proxy > 30 | GL_proxy 15-30 | GL_proxy < 15 |
|------------|---------------|----------------|---------------|
| est_meal | 1.0 | 0.7 | 0.4 |
| est_secondary_meal | 0.4 | 0.7 | 1.0 |

---

## Results: Today's Matches

| Logged Meal | Time | Matched Event | Event Time | Confidence |
|-------------|------|---------------|------------|------------|
| Pre Breakfast | 08:23 | est_secondary_meal | 07:02 | 80.6% |
| Breakfast | 10:57 | est_meal | 09:40 | 89.2% |
| Mid Morning Snack | 12:03 | est_secondary_meal | 10:17 | 71.0% |
| Lunch | 14:04 | est_meal | 14:11 | 89.7% |
| Afternoon Snack | 17:16 | est_secondary_meal | 16:37 | 99.7% |
| Dinner | 21:56 | est_meal | 20:56 | 91.5% |

---

## Key Insights

### Time Offset Patterns

| Meal | Time Offset (meal - detection) |
|------|-------------------------------|
| Pre Breakfast | +81 min |
| Breakfast | +77 min |
| Mid Morning Snack | +106 min |
| Lunch | -7 min |
| Afternoon Snack | +39 min |
| Dinner | +60 min |

**Observation:** Meals are typically logged 40-80 minutes AFTER the CGM detects the rise, except for Lunch which was logged almost exactly when detected. This suggests:
1. Users often log meals retrospectively
2. The detection algorithm is identifying the glucose inflection point early

### Physiological Validation

The **afternoon snack** had near-perfect matching (99.7%) because:
- Expected time-to-peak: 51 min
- Actual time-to-peak: 50 min (peak at 18:06, meal at 17:16)
- Event type (secondary) matched low GL_proxy (14.4)

---

## Assignment Algorithm

For optimal one-to-one matching, we use **greedy assignment**:

1. Compute all pairwise scores (meals × events)
2. Sort by composite score (descending)
3. Iterate: assign highest-scoring pair if neither meal nor event is already assigned
4. Continue until all meals are assigned (or no valid matches remain)

**Future improvement:** Hungarian algorithm for globally optimal assignment.

---

## Unmatched Events

The algorithm detected events that didn't match any logged meal:
- `est_meal @ 16:34` - possibly misclassified, or an unlogged snack
- `est_secondary_meal @ 14:17` - likely noise or continuation of lunch response

This suggests opportunities for:
1. Prompting users about potential unlogged meals
2. Refining the detection algorithm's sensitivity

---

## Code Reference

See `analyze_matching.py` for the full implementation.
