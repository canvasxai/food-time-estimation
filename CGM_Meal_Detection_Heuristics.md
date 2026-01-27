# CGM-Based Meal Timing Detection: Heuristics Guide

## Problem Statement

Users log meals in the app but the timing is inaccurate. We need to derive accurate meal timing from CGM (Continuous Glucose Monitor) data to create training data for predictive models.

**Current approach (Minima Method):** Find the minimum glucose value within a typical meal window (e.g., 6:30-9:30 PM for dinner) and assume that's the meal time. This works but only yields clear results for ~30% of samples.

**Goal:** Develop robust heuristics that work across diverse metabolic phenotypes (well-controlled individuals with glucose 75-120 mg/dL to poorly controlled diabetics ranging 150-400+ mg/dL).

---

## Physiological Background

Understanding why these heuristics work:

1. **Pre-prandial nadir:** Between meals, insulin levels drop and glucose gradually drifts toward a baseline low point
2. **Cephalic phase response:** Seeing/smelling food can trigger a slight insulin release, causing a small dip just before eating
3. **Postprandial rise:** Carbohydrate absorption causes glucose to rise 15-30 minutes after eating
4. **Peak timing:** Glucose typically peaks 60-90 minutes after meal consumption
5. **Return to baseline:** Glucose normalizes within 2-4 hours post-meal

---

## Heuristic 1: Rate-of-Change Inflection Point

### Concept
Instead of finding the absolute minimum, find where the derivative changes from negative/flat to sharply positive—the moment glucose *starts rising*.

### Algorithm
```
1. Calculate rolling rate of change (ROC) for each time interval
   ROC[t] = (glucose[t] - glucose[t-1]) / time_interval

2. Find inflection points where:
   - ROC[t-1] ≤ threshold_low (flat or declining)
   - ROC[t+1] ≥ threshold_high (rising)

3. Require sustained rise for 15-20 minutes to filter noise

4. Meal time = inflection point
```

### Critical: Individualized Thresholds

Fixed thresholds don't work across different metabolic phenotypes. Use one of these individualization methods:

#### Method A: Percentile-Based Thresholds
```
For each user:
1. Compute ROC for every interval across all their CGM history
2. Build distribution of ROC values
3. threshold_high = 85th-90th percentile of positive ROC values
4. threshold_low = median ROC or ~0
```

#### Method B: Coefficient of Variation Scaling
```
CV = std(glucose) / mean(glucose)
reference_CV = 0.15-0.20 (typical moderate control)

threshold_high = base_threshold × (user_CV / reference_CV)
```

#### Method C: Local Context Windowing
```
Within each meal window:
1. local_baseline = 25th percentile of glucose in that 3-hour window
2. local_ROC_std = standard deviation of ROC values in window
3. Flag inflection when ROC > local_mean_ROC + (k × local_ROC_std)
   where k = 1.5-2.0
```

### Pros
- Captures meals even without a clean minima
- Works when there's just a "bend" in the curve

### Cons
- Sensitive to noise; requires smoothing
- Threshold tuning needed per individual

---

## Heuristic 2: Peak Backtracking with Elbow Detection (Recommended)

### Concept
Work backwards from the postprandial peak to find where the rise began. Peaks are often more distinctive than minima.

### Algorithm
```
1. EXPAND SEARCH WINDOW
   - If meal window is 6:30-9:30 PM
   - Search for peaks in: 6:30 PM to 12:30 AM (window + 180 min)
   - This accounts for variable time-to-peak

2. FIND THE PEAK
   - Identify local maxima in the expanded window
   - Peak must be above threshold (e.g., >120 mg/dL or >20 mg/dL above recent baseline)

3. BACKTRACK TO FIND THE ELBOW
   - The "elbow" is where the curve transitions from flat to rising
   - This is the estimated meal time

4. Meal time = elbow point
```

### Elbow Detection Methods

#### Method A: Line-Distance Method (Recommended - Simple & Robust)
```
1. Define baseline point P1 = (t_baseline, glucose_baseline)
   - Located 90-120 minutes before peak

2. Define peak point P2 = (t_peak, glucose_peak)

3. Draw straight line from P1 to P2

4. For each CGM point between P1 and P2:
   - Calculate perpendicular distance to the line

5. Elbow = point with MAXIMUM distance from the line

Perpendicular distance formula:
distance = |((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)| / sqrt((y2-y1)² + (x2-x1)²)

Where (x0, y0) is the test point, (x1, y1) is P1, (x2, y2) is P2
```

#### Method B: Maximum Curvature (Second Derivative)
```
Curvature at point t = |g''(t)| / (1 + g'(t)²)^(3/2)

Steps:
1. Smooth the signal (rolling average or Savitzky-Golay filter)
2. Compute numerical first derivative g'(t)
3. Compute numerical second derivative g''(t)
4. Calculate curvature for each point
5. Elbow = point of maximum curvature between baseline and peak
```

#### Method C: Piecewise Linear Fit
```
Model the curve as two connected line segments: "flat then rising"

For each candidate breakpoint t_b between baseline and peak:
1. Fit line L1 to points from t_start to t_b
2. Fit line L2 to points from t_b to t_peak
3. Calculate total squared error for both segments

Elbow = breakpoint t_b that minimizes total squared error
```

### Edge Cases to Handle

**Multiple peaks (e.g., meal + dessert):**
- Option A: Find the FIRST significant peak, not the highest
- Option B: Find all peaks, run elbow detection on each, take earliest elbow as meal start

**No clear peak:**
- Fall back to Heuristic 1 (inflection detection) or Heuristic 3

### Pros
- Peaks are more distinctive than minima in noisy data
- Works even when pre-meal glucose was elevated
- Line-distance method is parameter-free

### Cons
- Requires sufficient post-meal data (extended window)
- May fail if meal response is very flat

---

## Heuristic 3: Baseline Deviation with Physiological Constraints

### Concept
Identify meal responses as statistically significant deviations from the individual's rolling baseline, validated by physiological timing rules.

### Algorithm
```
1. CALCULATE ROLLING BASELINE
   - Option A: 2-hour moving average
   - Option B: 25th percentile in a 3-hour sliding window

2. DETECT DEVIATIONS
   - Flag when glucose exceeds: baseline + threshold
   - Threshold = 15-20 mg/dL, or personalized based on user's variability

3. APPLY PHYSIOLOGICAL CONSTRAINTS
   All must be true for a valid meal detection:

   a) SUSTAINED RISE: Elevation lasts >20 minutes

   b) PLAUSIBLE RISE RATE:
      - Fast carbs: 2-4 mg/dL per minute
      - Slow carbs/mixed meals: 0.5-1.5 mg/dL per minute

   c) RETURN TO BASELINE: Glucose trends back down within 3-4 hours

4. Meal time = start of the sustained deviation
```

### Personalization
```
Calculate user's typical meal response amplitude:
typical_amplitude = median(peak - pre_meal_baseline) across known meals

Use this to calibrate deviation thresholds for that individual.
```

### Pros
- Robust to individual variation in fasting glucose levels
- Adapts to user's personal patterns
- Physiological constraints reduce false positives

### Cons
- Requires sufficient historical data to establish baseline patterns
- May miss small meal responses

---

## Heuristic 4: Original Minima Method (Reference)

### Algorithm
```
1. Define meal window (e.g., 6:30-9:30 PM for dinner)
2. Find the minimum glucose value within that window
3. Meal time = time of minimum
```

### When It Works
- Clean, well-defined glucose dip before meal
- Works well for ~30% of samples with clear patterns

### Limitations
- Fails when there's no distinct minima
- Fails with snacking or irregular eating patterns
- Fails with noisy CGM data

---

## Implementation Recommendations

### Recommended Approach: Composite Scoring

Use multiple heuristics and combine their outputs:

```
1. PRIMARY: Peak backtracking with elbow detection (Heuristic 2)
   - Most robust, works without clean minima

2. VALIDATION: Check if Rate-of-Change inflection (Heuristic 1) agrees
   - If elbow and inflection point are within 15-20 minutes, high confidence

3. FALLBACK: If Heuristic 2 fails (no clear peak), use:
   - Baseline deviation method (Heuristic 3), or
   - Original minima method (Heuristic 4)

4. CONFIDENCE SCORING:
   - 3+ heuristics agree: High confidence
   - 2 heuristics agree: Medium confidence
   - 1 heuristic only: Low confidence (flag for review)
```

### Data Preprocessing

Before applying any heuristic:
```
1. Handle missing data (interpolation or exclusion)
2. Apply smoothing to reduce noise:
   - Rolling average (5-15 minute window), or
   - Savitzky-Golay filter (preserves peaks better)
3. Ensure consistent time intervals
```

### Meal Windows (Typical)

| Meal      | Window Start | Window End | Peak Search Extension |
|-----------|--------------|------------|----------------------|
| Breakfast | 6:00 AM      | 10:00 AM   | +180 min (until 1:00 PM) |
| Lunch     | 11:00 AM     | 2:30 PM    | +180 min (until 5:30 PM) |
| Dinner    | 6:30 PM      | 9:30 PM    | +180 min (until 12:30 AM) |

Adjust windows based on user's reported patterns or regional norms.

### Output Format

For each detected meal, output:
```
{
  "date": "2024-10-11",
  "meal_type": "dinner",
  "detected_time": "2024-10-11T19:45:00",
  "detection_method": "peak_backtracking_elbow",
  "confidence": "high",
  "supporting_heuristics": ["inflection_point", "baseline_deviation"],
  "peak_time": "2024-10-11T20:45:00",
  "peak_value": 142,
  "pre_meal_baseline": 78,
  "rise_amplitude": 64
}
```

---

## Pseudocode: Peak Backtracking with Elbow Detection

```python
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def detect_meal_time(glucose, timestamps, meal_window_start, meal_window_end,
                     peak_extension_minutes=180):
    """
    Detect meal timing using peak backtracking with elbow detection.

    Parameters:
    -----------
    glucose : array-like
        CGM glucose values in mg/dL
    timestamps : array-like
        Corresponding timestamps
    meal_window_start : datetime
        Start of expected meal window
    meal_window_end : datetime
        End of expected meal window
    peak_extension_minutes : int
        How far past meal_window_end to search for peaks

    Returns:
    --------
    dict with detected meal time, confidence, and supporting data
    """

    # Step 1: Smooth the signal
    glucose_smooth = savgol_filter(glucose, window_length=7, polyorder=2)

    # Step 2: Define peak search window (meal window + extension)
    peak_search_end = meal_window_end + timedelta(minutes=peak_extension_minutes)

    # Step 3: Find peaks in extended window
    # Mask to extended window
    mask = (timestamps >= meal_window_start) & (timestamps <= peak_search_end)
    window_glucose = glucose_smooth[mask]
    window_times = timestamps[mask]

    # Find peaks with minimum prominence
    peaks, properties = find_peaks(window_glucose,
                                   prominence=20,  # Adjust based on user's typical response
                                   distance=12)    # At least 1 hour apart (if 5-min intervals)

    if len(peaks) == 0:
        return {"success": False, "reason": "No peaks found"}

    # Take the first significant peak
    peak_idx = peaks[0]
    peak_time = window_times[peak_idx]
    peak_value = window_glucose[peak_idx]

    # Step 4: Backtrack to find elbow
    # Look back 120 minutes from peak
    lookback_minutes = 120
    samples_per_minute = 1 / 5  # Assuming 5-minute intervals
    lookback_samples = int(lookback_minutes * samples_per_minute)

    baseline_idx = max(0, peak_idx - lookback_samples)

    # Find elbow using line-distance method
    elbow_idx = find_elbow_line_distance(
        glucose=window_glucose[baseline_idx:peak_idx+1],
        start_idx=0,
        end_idx=peak_idx - baseline_idx
    )

    # Convert back to original indices
    elbow_idx_absolute = baseline_idx + elbow_idx
    meal_time = window_times[elbow_idx_absolute]

    return {
        "success": True,
        "meal_time": meal_time,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "pre_meal_glucose": window_glucose[elbow_idx_absolute],
        "method": "peak_backtracking_elbow"
    }


def find_elbow_line_distance(glucose, start_idx, end_idx):
    """
    Find elbow point using perpendicular distance to line method.

    The elbow is the point farthest from the straight line connecting
    the start point to the end point.
    """
    # Normalize x-axis to [0, 1] for numerical stability
    n_points = end_idx - start_idx + 1
    x = np.linspace(0, 1, n_points)
    y = glucose[start_idx:end_idx+1]

    # Line from start to end: y = mx + b
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # Calculate perpendicular distance for each point
    # Using formula: |Ax + By + C| / sqrt(A² + B²)
    # Where line is: (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0

    A = y2 - y1
    B = -(x2 - x1)
    C = x2 * y1 - y2 * x1

    distances = np.abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)

    # Elbow is point with maximum distance
    elbow_idx = np.argmax(distances)

    return start_idx + elbow_idx


def calculate_confidence(primary_meal_time, heuristic_results):
    """
    Calculate confidence score based on agreement between heuristics.

    Parameters:
    -----------
    primary_meal_time : datetime
        Meal time from primary heuristic
    heuristic_results : list of dict
        Results from other heuristics, each with 'meal_time' key

    Returns:
    --------
    str: 'high', 'medium', or 'low'
    """
    agreement_threshold_minutes = 20
    agreeing_heuristics = 0

    for result in heuristic_results:
        if result.get('success', False):
            time_diff = abs((result['meal_time'] - primary_meal_time).total_seconds() / 60)
            if time_diff <= agreement_threshold_minutes:
                agreeing_heuristics += 1

    if agreeing_heuristics >= 2:
        return 'high'
    elif agreeing_heuristics >= 1:
        return 'medium'
    else:
        return 'low'
```

---

## Testing & Validation

### Validation Approach
1. Use the 30% of samples where minima method worked (ground truth-ish)
2. Compare new heuristic results against minima-detected times
3. Measure agreement within ±15-20 minute tolerance

### Metrics to Track
- Detection rate: % of samples where a meal time was detected
- Accuracy: % of detected times within ±20 min of reference
- Precision by confidence level: Accuracy for high/medium/low confidence detections

### Edge Cases to Test
- Flat glucose profiles (very low carb meals)
- Multiple peaks (meal + snack)
- Missing data gaps
- Very high glucose individuals (>300 mg/dL baseline)
- Very stable individuals (glucose SD < 10 mg/dL)

---

## References

- Postprandial glucose dynamics in CGM literature
- Elbow method (Thorndike, 1953) adapted from clustering applications
- Savitzky-Golay filter for signal smoothing
