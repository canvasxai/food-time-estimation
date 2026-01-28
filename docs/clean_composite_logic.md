# Clean vs Composite Event Classification Logic

## Overview

This document describes the logic used to classify CGM meal events as either **Clean** or **Composite** based on their relationship to glucose peaks.

## Definitions

### Clean Events
A meal event is **CLEAN** when it has a direct relationship to its peak without interference from other distinct meals:

1. **`to_peak`** - Meal event → Peak
   - The next thing after the meal event is a glucose peak
   - No other meal events occur before the peak

2. **`to_peak_merged`** - Meal event → (Merged meal) → Peak
   - There's another meal event before the peak, BUT it's within the `merge_gap_minutes` (default 30 min)
   - Meals within the merge gap are considered part of the same eating event
   - Example: Breakfast at 8:00 and a secondary meal at 8:20 → both treated as one clean event

### Composite Events
A meal event is **COMPOSITE** when multiple distinct meals share the same glucose peak:

1. **`to_next_meal`** - Meal event → Meal event → Peak
   - The next thing after the meal event is another distinct meal (outside merge gap)
   - Both meals eventually lead to the same peak
   - ALL meals sharing that peak are marked composite

## Classification Rules

| Scenario | Time Gap | Classification |
|----------|----------|---------------|
| Meal → Peak (no intervening meals) | N/A | Clean |
| Meal → Meal → Peak | ≤ 30 min | Clean (merged) |
| Meal → Meal → Peak | > 30 min | Composite (both meals) |

## Example Timeline

```
08:00  Meal A (MEAL_START)
08:20  Meal B (SECONDARY_MEAL)  ← within 30 min of A
09:30  Peak
10:00  Meal C (MEAL_START)
11:30  Meal D (SECONDARY_MEAL)  ← 90 min after C
12:00  Peak

Result:
- Meal A: Clean (to_peak_merged - merged with B, within merge gap)
- Meal B: Clean (to_peak_merged - merged with A, within merge gap)
- Meal C: Composite (to_next_meal - D is >30 min away, shares peak)
- Meal D: Composite (to_peak but shares peak with C)
```

### Merge Group Detection

An event is considered "merged" if it's within `merge_gap_minutes` of **either** its previous or next neighbor:

```
is_merged_with_next = time_to_next_meal <= merge_gap_minutes
is_merged_with_prev = time_from_prev_meal <= merge_gap_minutes
is_part_of_merge_group = is_merged_with_next OR is_merged_with_prev
```

This ensures that **both events in a merged pair receive the same `to_peak_merged` classification**.

## Algorithm Steps

### Step 1: Build Segments
For each meal event, determine:
- Next meal event (if any)
- Next peak after this event
- Time gap to next meal
- Whether next meal is within merge gap

### Step 2: Classify Segments
```
For each segment:
    If segment_type == "to_next_meal":
        → COMPOSITE (distinct meal before peak)
    Else if segment_type in ("to_peak", "to_peak_merged"):
        → Check if previous event shares same peak AND was "to_next_meal"
        → If yes: COMPOSITE (propagate status)
        → If no: CLEAN
```

### Step 3: Propagate Composite Status
When an event going `to_peak` shares the same peak with a previous `to_next_meal` event:
- Mark current event as composite
- Walk backwards and mark all events sharing that peak as composite

## Implementation Locations

| File | Function | Purpose |
|------|----------|---------|
| `detection.py` | `detect_composite_events()` | Creates CompositeMealEvent objects with clean/composite classification |
| `matching.py` | `match_meals_to_events()` | Applies same logic when matching meals to CGM events |

## Matching Behavior

### How Merged Events Are Handled

**Current Implementation:**
- `detect_meal_events_simplified()` merges nearby events using `merge_nearby_meal_events()`
- Events within `event_merge_gap_minutes` (default 30 min) are merged into a single event
- The merged list is passed to `match_meals_to_events()`

**Important Note:**
The matching function (`match_meals_to_events`) re-detects fresh secondary meals from CGM data (lines 93-99):
```python
if cgm_df is not None and thresholds is not None:
    fresh_secondary = detect_secondary_meal_events(cgm_df, thresholds)
    existing_times = {e.detected_at for e in meal_events}
    for sec in fresh_secondary:
        if sec.detected_at not in existing_times:
            meal_events.append(sec)
```

This means:
1. **Merged events from detection** are passed in as a single event
2. **Fresh secondary detection** may re-add events that were previously merged
3. The `merge_gap_minutes` check in matching ensures these re-added events are still classified as **clean** (not composite)

### Matching Process

1. **Collect Events**: Get MEAL_START and SECONDARY_MEAL events, plus fresh secondary meals
2. **Sort by Time**: All meal events sorted chronologically
3. **Build Segments**: For each event, determine segment type considering merge gap
4. **Classify**: Apply clean/composite logic based on segment type
5. **Match to Logged Meals**: Use slot ordering and carbs/ΔG ranking

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `event_merge_gap_minutes` | 30 | Events within this gap are considered merged (clean) |

This can be adjusted in the Streamlit sidebar under "Event Merge Gap".

## Visual Indicators

In the visualization:
- **Clean events**: Labeled "Clean" in green (#2ECC71)
- **Composite events**: Labeled "Composite" in purple (#9B59B6)
- **Composite events** also show `→+XX` indicating ΔG to the next event
