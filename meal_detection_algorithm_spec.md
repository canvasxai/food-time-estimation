# Meal Detection Algorithm Specification
## Derivative-Based CGM Analysis for Automatic Meal Time Inference

---

## 1. Overview

This document specifies an algorithm to detect meal events from continuous glucose monitor (CGM) data using derivative analysis. The goal is to infer when a user ate, even if they didn't log their meals accurately, by analyzing the glucose curve's rate of change patterns.

### Core Principle
- **Peaks** tell us "something ended" (absorption complete)
- **First derivative (dG/dt)** tells us "something started" (meal onset)
- **Second derivative (d²G/dt²)** reveals hidden events (snacks, stacked meals)

---

## 2. Definitions and Calculations

### 2.1 Input Data
- Time series of glucose readings (typically every 5 minutes)
- Format: `[(timestamp, glucose_mg_dL), ...]`

### 2.2 Preprocessing

#### 2.2.1 Smoothing
Apply a rolling average to reduce sensor noise before derivative calculation.

```
smoothed_glucose[t] = mean(glucose[t-2 : t+2])  # 5-point moving average
```

**Rationale**: CGM sensors have measurement noise of ±10-15 mg/dL. Smoothing prevents false derivative spikes.

#### 2.2.2 First Derivative (Rate of Change)
Calculate dG/dt at each point:

```
dG_dt[t] = (smoothed_glucose[t+1] - smoothed_glucose[t-1]) / (time[t+1] - time[t-1])
```

**Units**: mg/dL per minute

#### 2.2.3 Second Derivative (Acceleration)
Calculate d²G/dt² at each point:

```
d2G_dt2[t] = (dG_dt[t+1] - dG_dt[t-1]) / (time[t+1] - time[t-1])
```

**Units**: mg/dL per minute²

---

## 3. Event Detection Logic

### 3.1 Event Types

The algorithm detects four types of events:

| Event Type | Description | Typical Scenario |
|------------|-------------|------------------|
| **MEAL_CLEAN** | New meal on a stable/declining baseline | Breakfast after fasting |
| **MEAL_STACKED** | Second meal before first has cleared | Dinner shortly after snack |
| **SNACK_HIDDEN** | Small intake that slows decline without causing rise | Nuts during post-meal decline |
| **PEAK** | Absorption complete, decline begins | 60-90 min post-meal |

### 3.2 Detection Rules

#### 3.2.1 MEAL_CLEAN (New Meal on Clean Baseline)

**Condition**:
```
IF:
    dG_dt[t-1] < -0.3 mg/dL/min (was declining)
    AND dG_dt[t] crosses above +0.5 mg/dL/min (now rising)
    AND this rise sustains for >= 15 minutes
THEN:
    FLAG: MEAL_CLEAN at time t
    Estimated meal time: t minus 10-15 minutes (absorption lag)
```

**Rationale**: A clean crossover from negative to positive derivative, sustained over time, indicates new glucose entering the bloodstream.

#### 3.2.2 MEAL_STACKED (Overlapping Meal During Active Absorption)

**Condition**:
```
IF:
    dG_dt[t] > 0 (already rising, previous meal in progress)
    AND d2G_dt2[t] spikes above +0.05 mg/dL/min² (re-acceleration)
    AND this acceleration sustains for >= 10 minutes
THEN:
    FLAG: MEAL_STACKED at time t
    Estimated meal time: t minus 10-15 minutes
```

**Rationale**: If glucose is already rising and suddenly accelerates further, a second food intake has occurred. This is the "stacked meal" scenario (e.g., snack at 18:57, then dinner at 20:18).

#### 3.2.3 SNACK_HIDDEN (Hidden Snack During Decline)

**Condition**:
```
IF:
    dG_dt[t-1] < -1.0 mg/dL/min (was declining steeply)
    AND dG_dt[t] is still negative but > -0.5 mg/dL/min (decline slowed significantly)
    AND d2G_dt2[t] > +0.03 mg/dL/min² (positive acceleration while still falling)
    AND glucose does NOT cross into positive dG_dt within 20 minutes
THEN:
    FLAG: SNACK_HIDDEN at time t
    Estimated snack time: t minus 10-15 minutes
```

**Rationale**: A snack with low carbs won't reverse the decline but will cushion it. The derivative stays negative but becomes less negative, showing as positive second derivative.

#### 3.2.4 PEAK (Absorption Complete)

**Condition**:
```
IF:
    dG_dt[t-1] > +0.3 mg/dL/min (was rising)
    AND dG_dt[t] crosses below -0.3 mg/dL/min (now declining)
    AND glucose[t] is a local maximum (higher than t-2 and t+2)
THEN:
    FLAG: PEAK at time t
```

**Rationale**: Peaks mark the end of absorption. They confirm a meal occurred ~60-90 minutes prior.

---

## 4. Special Handling: Post-Dawn Effect (7:00 - 9:00 AM)

### 4.1 The Problem

The dawn effect causes glucose to rise overnight (typically 4:00-7:00 AM) due to cortisol and growth hormone release. After 7:00 AM, this effect wanes, creating a natural decline that can confuse meal detection.

**Observed Pattern (7:00-9:00 AM)**:
- Initial decline as dawn effect clears
- Possible stabilization or minor fluctuation
- Then actual breakfast causes a rise

### 4.2 Post-Dawn Algorithm Adjustments

#### Phase 1: Dawn Clearance (7:00 - 8:00 AM typical)

```
DURING dawn_clearance_phase:
    - EXPECT negative dG_dt (natural decline)
    - DO NOT flag positive d2G_dt2 as SNACK_HIDDEN
    - This is normal homeostatic correction, not a hidden meal
```

#### Phase 2: Stabilization Detection

```
DETECT stabilization WHEN:
    |dG_dt| < 0.3 mg/dL/min for >= 10 consecutive minutes
    OR glucose reaches a local minimum (dG_dt crosses from - to +)

ONCE stabilized:
    - Exit dawn_clearance_phase
    - Begin normal meal detection BUT with elevated thresholds
```

#### Phase 3: First Meal Detection (Higher Threshold)

```
FOR first meal after dawn:
    REQUIRE:
        dG_dt > +1.0 mg/dL/min (higher than normal 0.5 threshold)
        AND sustained for >= 20 minutes (longer than normal 15 min)

    Rationale: Avoid false positives from minor post-dawn fluctuations
```

#### Phase 4: Return to Normal Sensitivity

```
AFTER first confirmed meal (breakfast):
    - Revert to standard thresholds for all subsequent events
    - Normal detection rules apply for rest of day
```

### 4.3 Dawn Effect Time Window

```
DEFAULT dawn_effect_window: 7:00 AM - 9:00 AM

CONFIGURABLE: Some users may have earlier/later dawn patterns
    - Can be calibrated from multi-day CGM data
    - Look for consistent overnight rise pattern
```

---

## 5. Threshold Configuration

### 5.1 Default Thresholds

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `smoothing_window` | 5 points | Rolling average window size |
| `dGdt_rising_threshold` | +0.5 mg/dL/min | Minimum rate to count as "rising" |
| `dGdt_falling_threshold` | -0.3 mg/dL/min | Rate to count as "falling" |
| `d2Gdt2_acceleration_threshold` | +0.05 mg/dL/min² | Minimum acceleration for stacked meal |
| `d2Gdt2_snack_threshold` | +0.03 mg/dL/min² | Minimum acceleration for hidden snack |
| `min_sustain_duration` | 15 minutes | How long a rise must sustain |
| `meal_absorption_lag` | 10-15 minutes | Time from eating to detectable rise |
| `post_dawn_dGdt_threshold` | +1.0 mg/dL/min | Higher threshold for first morning meal |
| `post_dawn_sustain_duration` | 20 minutes | Longer sustain for first morning meal |
| `dawn_window_start` | 07:00 | Start of post-dawn handling |
| `dawn_window_end` | 09:00 | End of post-dawn handling |

### 5.2 Calibration Notes

These thresholds should be calibrated per-user if possible:
- Users with insulin resistance may have slower rises (lower dG/dt)
- Users on medication may have dampened responses
- Athletes may have faster clearance (steeper negative dG/dt)

---

## 6. Algorithm Pseudocode

```python
def detect_meal_events(cgm_data):
    """
    Main entry point for meal detection.

    Args:
        cgm_data: List of (timestamp, glucose_mg_dL) tuples

    Returns:
        List of detected events with type, time, and confidence
    """

    # Step 1: Preprocess
    smoothed = apply_rolling_average(cgm_data, window=5)
    dG_dt = calculate_first_derivative(smoothed)
    d2G_dt2 = calculate_second_derivative(dG_dt)

    events = []
    in_dawn_phase = True
    dawn_stabilized = False
    first_meal_detected = False

    for t in range(len(cgm_data)):
        timestamp = cgm_data[t].timestamp

        # Step 2: Handle dawn effect window
        if is_in_dawn_window(timestamp):
            if not dawn_stabilized:
                if is_stabilized(dG_dt, t):
                    dawn_stabilized = True
                else:
                    continue  # Skip detection during dawn clearance

            # Use elevated thresholds for first meal
            if not first_meal_detected:
                thresholds = POST_DAWN_THRESHOLDS
            else:
                thresholds = NORMAL_THRESHOLDS
        else:
            in_dawn_phase = False
            thresholds = NORMAL_THRESHOLDS

        # Step 3: Check for PEAK
        if is_peak(dG_dt, smoothed, t):
            events.append(Event(type="PEAK", time=timestamp))

        # Step 4: Check for MEAL_CLEAN
        if is_meal_clean(dG_dt, t, thresholds):
            estimated_meal_time = timestamp - ABSORPTION_LAG
            events.append(Event(type="MEAL_CLEAN", time=estimated_meal_time))
            first_meal_detected = True

        # Step 5: Check for MEAL_STACKED
        if is_meal_stacked(dG_dt, d2G_dt2, t, thresholds):
            estimated_meal_time = timestamp - ABSORPTION_LAG
            events.append(Event(type="MEAL_STACKED", time=estimated_meal_time))

        # Step 6: Check for SNACK_HIDDEN (skip during dawn phase)
        if not in_dawn_phase:
            if is_snack_hidden(dG_dt, d2G_dt2, t, thresholds):
                estimated_snack_time = timestamp - ABSORPTION_LAG
                events.append(Event(type="SNACK_HIDDEN", time=estimated_snack_time))

    # Step 7: Post-process to merge nearby events
    events = merge_nearby_events(events, min_gap=20)  # 20 minute minimum gap

    return events


def is_meal_clean(dG_dt, t, thresholds):
    """Detect crossover from falling to rising."""
    if t < 2:
        return False

    was_falling = dG_dt[t-1] < thresholds.falling_threshold
    now_rising = dG_dt[t] > thresholds.rising_threshold

    if was_falling and now_rising:
        # Check if rise sustains
        return check_sustained_rise(dG_dt, t, thresholds.min_sustain_duration)

    return False


def is_meal_stacked(dG_dt, d2G_dt2, t, thresholds):
    """Detect re-acceleration during active rise."""
    if t < 2:
        return False

    already_rising = dG_dt[t] > 0
    accelerating = d2G_dt2[t] > thresholds.acceleration_threshold

    if already_rising and accelerating:
        # Check if acceleration sustains
        return check_sustained_acceleration(d2G_dt2, t, duration=10)

    return False


def is_snack_hidden(dG_dt, d2G_dt2, t, thresholds):
    """Detect cushioned decline (snack during fall)."""
    if t < 2:
        return False

    was_falling_steeply = dG_dt[t-1] < -1.0
    still_falling_but_slower = -0.5 < dG_dt[t] < 0
    positive_acceleration = d2G_dt2[t] > thresholds.snack_threshold

    if was_falling_steeply and still_falling_but_slower and positive_acceleration:
        # Verify it doesn't turn into a full rise (that would be MEAL_CLEAN)
        return not becomes_positive_soon(dG_dt, t, lookahead=20)

    return False


def is_peak(dG_dt, glucose, t):
    """Detect local maximum with derivative crossover."""
    if t < 2 or t >= len(dG_dt) - 2:
        return False

    was_rising = dG_dt[t-1] > 0.3
    now_falling = dG_dt[t] < -0.3
    is_local_max = glucose[t] > glucose[t-2] and glucose[t] > glucose[t+2]

    return was_rising and now_falling and is_local_max
```

---

## 7. Output Format

### 7.1 Event Structure

```python
class MealEvent:
    event_type: str      # "MEAL_CLEAN", "MEAL_STACKED", "SNACK_HIDDEN", "PEAK"
    detected_at: datetime    # When the signal was detected in CGM data
    estimated_meal_time: datetime  # Back-calculated actual eating time
    confidence: float    # 0.0 - 1.0 based on signal strength
    dG_dt_at_detection: float    # First derivative value
    d2G_dt2_at_detection: float  # Second derivative value
    glucose_at_detection: float  # Glucose value at detection point
```

### 7.2 Example Output

```json
{
  "events": [
    {
      "event_type": "MEAL_CLEAN",
      "detected_at": "2024-01-15T09:15:00",
      "estimated_meal_time": "2024-01-15T09:00:00",
      "confidence": 0.92,
      "dG_dt_at_detection": 1.8,
      "d2G_dt2_at_detection": 0.12,
      "glucose_at_detection": 118
    },
    {
      "event_type": "PEAK",
      "detected_at": "2024-01-15T10:05:00",
      "estimated_meal_time": null,
      "confidence": 0.95,
      "dG_dt_at_detection": -0.8,
      "d2G_dt2_at_detection": -0.05,
      "glucose_at_detection": 191
    }
  ]
}
```

---

## 8. Validation Against Ground Truth

### 8.1 Test Data

Use the provided labeled dataset for validation:

| Actual Time | Meal | Carbs (g) | Expected Detection |
|-------------|------|-----------|-------------------|
| 09:01 | Breakfast | 37.3 | MEAL_CLEAN |
| 13:44 | Lunch | 36.8 | MEAL_CLEAN |
| 16:13 | Snack | 0.4 | Likely undetectable (too low carb) |
| 18:57 | Fish Fry | 3.1 | Possibly SNACK_HIDDEN or undetectable |
| 20:18 | Dinner | 31.9 | MEAL_STACKED (if fish fry detected) or MEAL_CLEAN |

### 8.2 Success Metrics

```
True Positive: Detected event within ±30 minutes of actual meal
False Positive: Detected event with no corresponding meal
False Negative: Missed actual meal (>15g carbs)

Target Performance:
- Sensitivity (recall): >85% for meals with >20g carbs
- Precision: >75% (allow some false positives from ambiguous signals)
- Time accuracy: ±20 minutes of actual meal time
```

### 8.3 Known Limitations

1. **Very low carb meals** (<5g) may be undetectable
2. **High-fat meals** cause delayed, prolonged rises that may confuse timing
3. **Exercise** can cause glucose drops that mimic or mask meal patterns
4. **Medications** (insulin, metformin) alter the curve shape
5. **Individual variation** means thresholds need per-user calibration

---

## 9. Implementation Notes

### 9.1 Data Requirements

- Minimum CGM sampling: every 5 minutes
- Minimum analysis window: 2 hours (to establish baseline patterns)
- Recommended: full 24-hour data for best accuracy

### 9.2 Edge Cases to Handle

1. **Missing data points**: Interpolate if gap < 15 minutes, otherwise mark segment as unreliable
2. **Sensor compression lows**: CGM can show false lows when lying on sensor; filter outliers
3. **Rapid consecutive meals**: If two meals within 30 minutes, may appear as single event
4. **Overnight eating**: Apply different logic for meals between 10 PM - 6 AM (less common)

### 9.3 Visualization Recommendations

Generate plots showing:
1. Raw glucose curve
2. Smoothed glucose curve
3. First derivative (dG/dt) subplot
4. Second derivative (d²G/dt²) subplot
5. Vertical markers for detected events (color-coded by type)
6. Ground truth meal times (if available) for comparison

---

## 10. Future Enhancements

1. **Machine learning refinement**: Train classifier on labeled data to improve thresholds
2. **Meal size estimation**: Correlate area-under-curve with carbohydrate content
3. **Meal type inference**: Different rise patterns for fast vs. slow carbs
4. **Activity integration**: Incorporate accelerometer data to filter exercise effects
5. **Multi-day learning**: Calibrate user-specific thresholds from historical patterns

---

## Appendix A: Quick Reference - Detection Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEAL DETECTION QUICK REFERENCE              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MEAL_CLEAN (Fresh meal on stable baseline)                     │
│  ─────────────────────────────────────────                      │
│  Signal: dG/dt crosses from negative to positive                │
│  Threshold: dG/dt > +0.5 mg/dL/min, sustained 15+ min           │
│                                                                 │
│  MEAL_STACKED (Second meal during active absorption)            │
│  ─────────────────────────────────────────────────              │
│  Signal: d²G/dt² spikes positive while dG/dt already positive   │
│  Threshold: d²G/dt² > +0.05 mg/dL/min², sustained 10+ min       │
│                                                                 │
│  SNACK_HIDDEN (Small intake cushioning decline)                 │
│  ─────────────────────────────────────────────                  │
│  Signal: Decline slows but doesn't reverse                      │
│  Threshold: d²G/dt² > +0.03, dG/dt stays negative               │
│                                                                 │
│  PEAK (Absorption complete)                                     │
│  ─────────────────────────                                      │
│  Signal: dG/dt crosses from positive to negative                │
│  Threshold: Local maximum in glucose values                     │
│                                                                 │
│  POST-DAWN HANDLING (7:00 - 9:00 AM)                            │
│  ───────────────────────────────────                            │
│  - Wait for stabilization before detection                      │
│  - Use elevated thresholds for first meal                       │
│  - Ignore SNACK_HIDDEN during dawn clearance                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0*
*Created: January 2026*
*Purpose: Specification for automated meal detection from CGM data*
