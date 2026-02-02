"""
Visualization functions for CGM data and meal detection.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from .models import MealEvent, MealMatch, SimplifiedThresholds, CompositeMealEvent
from .derivatives import apply_smoothing, calculate_first_derivative, calculate_second_derivative
from .detection import detect_secondary_meal_events


def create_simplified_derivative_plot(cgm_df: pd.DataFrame, events: List[MealEvent],
                                       thresholds: SimplifiedThresholds,
                                       matches: List[MealMatch] = None,
                                       derived_meal_times: List = None,
                                       composite_events: List[CompositeMealEvent] = None,
                                       discarded_events: List[MealEvent] = None,
                                       dietary_response_df: pd.DataFrame = None) -> go.Figure:
    """
    Create a plot for the simplified detection method showing:
    1. Raw and smoothed glucose curves
    2. First derivative (dG/dt) with zero line and threshold bands
    3. Second derivative (d²G/dt²)
    4. Detected events marked
    5. Matched meal events annotated (if matches provided)
    6. Merged meal events with mean time markers (if composite_events provided)
    7. Discarded events shown in grey (if discarded_events provided)
    """
    df = cgm_df.sort_values("datetime_ist").copy().reset_index(drop=True)

    # Apply preprocessing
    df["smoothed"] = apply_smoothing(df, window=thresholds.smoothing_window)
    df["dG_dt"] = calculate_first_derivative(df)
    df["d2G_dt2"] = calculate_second_derivative(df)

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "CGM Glucose (Raw & Smoothed)",
            "First Derivative (dG/dt) - Zero Crossings Indicate Events",
            "Second Derivative (d²G/dt²) - Acceleration"
        ),
        row_heights=[0.45, 0.30, 0.25]
    )

    # Plot 1: Raw glucose
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["value"],
            mode="lines+markers",
            name="CGM (raw)",
            line=dict(color="#2E86AB", width=2),
            marker=dict(size=5),
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose (raw):</b> %{y:.1f} mg/dL<extra></extra>"
        ),
        row=1, col=1
    )

    # Plot 1: Smoothed glucose
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["smoothed"],
            mode="lines",
            name="CGM (smoothed)",
            line=dict(color="#E94F37", width=1.5, dash="dot"),
            opacity=0.7,
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose (smoothed):</b> %{y:.1f} mg/dL<extra></extra>"
        ),
        row=1, col=1
    )

    # Get primary meal events from the events list passed in
    primary_events = [e for e in events if e.event_type == "MEAL_START"]

    # Get secondary meal events from detection module (single source of truth)
    passing_events, filtered_events, merged_events = detect_secondary_meal_events(
        cgm_df, thresholds, return_filtered=True, primary_events=primary_events
    )

    # Plot merged peaks (orange) - peaks that will be merged with a MEAL_START
    if merged_events:
        merged_times = [e.detected_at for e in merged_events]
        merged_glucose = [e.glucose_at_detection for e in merged_events]
        merged_d2G = [e.d2G_dt2_at_detection for e in merged_events]
        merged_dG = [e.dG_dt_at_detection for e in merged_events]

        fig.add_trace(
            go.Scatter(
                x=merged_times,
                y=merged_glucose,
                mode="markers",
                name="d²G/dt² Peak (merged)",
                marker=dict(
                    size=10,
                    color="#E67E22",
                    symbol="diamond-open",
                    line=dict(color="#E67E22", width=2)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<br><b>dG/dt:</b> %{customdata[0]:.4f}<br><b>d²G/dt²:</b> %{customdata[1]:.4f}<br><b>Status:</b> Merged with nearby MEAL_START<extra></extra>",
                customdata=list(zip(merged_dG, merged_d2G))
            ),
            row=1, col=1
        )

        # Add markers on d²G/dt² plot for merged peaks
        fig.add_trace(
            go.Scatter(
                x=merged_times,
                y=merged_d2G,
                mode="markers",
                name="d²G/dt² Peak (merged)",
                marker=dict(
                    size=8,
                    color="#E67E22",
                    symbol="diamond-open",
                    line=dict(color="#E67E22", width=2)
                ),
                showlegend=False,
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>d²G/dt²:</b> %{y:.4f} mg/dL/min²<br><b>Status:</b> Merged<extra></extra>"
            ),
            row=3, col=1
        )

    # Plot filtered peaks (gray) - peaks that don't pass dG/dt threshold
    if filtered_events:
        filtered_times = [e.detected_at for e in filtered_events]
        filtered_glucose = [e.glucose_at_detection for e in filtered_events]
        filtered_d2G = [e.d2G_dt2_at_detection for e in filtered_events]
        filtered_dG = [e.dG_dt_at_detection for e in filtered_events]

        fig.add_trace(
            go.Scatter(
                x=filtered_times,
                y=filtered_glucose,
                mode="markers",
                name="d²G/dt² Peak (filtered)",
                marker=dict(
                    size=10,
                    color="#95A5A6",
                    symbol="diamond-open",
                    line=dict(color="#95A5A6", width=2)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<br><b>dG/dt:</b> %{customdata[0]:.4f}<br><b>d²G/dt²:</b> %{customdata[1]:.4f}<br><b>Status:</b> Filtered (dG/dt below threshold)<extra></extra>",
                customdata=list(zip(filtered_dG, filtered_d2G))
            ),
            row=1, col=1
        )

        # Add markers on d²G/dt² plot for filtered peaks
        fig.add_trace(
            go.Scatter(
                x=filtered_times,
                y=filtered_d2G,
                mode="markers",
                name="d²G/dt² Peak (filtered)",
                marker=dict(
                    size=8,
                    color="#95A5A6",
                    symbol="diamond-open",
                    line=dict(color="#95A5A6", width=2)
                ),
                showlegend=False,
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>d²G/dt²:</b> %{y:.4f} mg/dL/min²<br><b>Status:</b> Filtered<extra></extra>"
            ),
            row=3, col=1
        )

    # Plot discarded meal events (grey) - events with insufficient ΔG
    if discarded_events:
        discarded_times = [e.detected_at for e in discarded_events]
        discarded_glucose = [e.glucose_at_detection for e in discarded_events]
        discarded_dG = [e.dG_dt_at_detection for e in discarded_events]

        fig.add_trace(
            go.Scatter(
                x=discarded_times,
                y=discarded_glucose,
                mode="markers",
                name="Discarded (insufficient ΔG)",
                marker=dict(
                    size=10,
                    color="#95A5A6",
                    symbol="triangle-up-open",
                    line=dict(color="#95A5A6", width=2)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<br><b>dG/dt:</b> %{customdata:.4f}<br><b>Status:</b> Discarded (ΔG < 5 mg/dL)<extra></extra>",
                customdata=discarded_dG
            ),
            row=1, col=1
        )

        # Add grey vertical lines on dG/dt plot for discarded events
        for event in discarded_events:
            event_time_ms = int(event.detected_at.timestamp() * 1000)
            fig.add_vline(
                x=event_time_ms,
                line_width=1,
                line_dash="dash",
                line_color="#95A5A6",
                opacity=0.5,
                row=2, col=1
            )

    # Plot passing peaks (green) - peaks that pass dG/dt threshold
    # Only shown on d²G/dt² plot (row 3), removed from glucose plot to reduce clutter
    if passing_events:
        passing_times = [e.detected_at for e in passing_events]
        passing_d2G = [e.d2G_dt2_at_detection for e in passing_events]

        # Add markers on d²G/dt² plot only
        fig.add_trace(
            go.Scatter(
                x=passing_times,
                y=passing_d2G,
                mode="markers",
                name="Est. Secondary Meal",
                marker=dict(
                    size=10,
                    color="#27AE60",
                    symbol="diamond",
                    line=dict(color="white", width=1)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>d²G/dt²:</b> %{y:.4f} mg/dL/min²<extra>Est. Secondary Meal</extra>"
            ),
            row=3, col=1
        )

    # Plot 2: First derivative
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["dG_dt"],
            mode="lines",
            name="dG/dt",
            line=dict(color="#28A745", width=2),
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>dG/dt:</b> %{y:.3f} mg/dL/min<extra></extra>"
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2, row=2, col=1)

    # Add threshold bands
    fig.add_hline(y=thresholds.min_derivative_magnitude, line_dash="dash",
                  line_color="gray", line_width=1, row=2, col=1,
                  annotation_text=f"+{thresholds.min_derivative_magnitude}")
    fig.add_hline(y=-thresholds.min_derivative_magnitude, line_dash="dash",
                  line_color="gray", line_width=1, row=2, col=1,
                  annotation_text=f"-{thresholds.min_derivative_magnitude}")

    # Add shaded "noise zone"
    x_min = df["datetime_ist"].min()
    x_max = df["datetime_ist"].max()

    fig.add_shape(
        type="rect",
        x0=x_min, x1=x_max,
        y0=-thresholds.min_derivative_magnitude,
        y1=thresholds.min_derivative_magnitude,
        fillcolor="lightgray",
        opacity=0.3,
        line_width=0,
        row=2, col=1
    )

    # Plot 3: Second derivative
    fig.add_trace(
        go.Scatter(
            x=df["datetime_ist"],
            y=df["d2G_dt2"],
            mode="lines",
            name="d²G/dt²",
            line=dict(color="#FF6B35", width=2),
            hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>d²G/dt²:</b> %{y:.4f} mg/dL/min²<extra></extra>"
        ),
        row=3, col=1
    )

    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=3, col=1)

    # Add start hour vertical line
    start_datetime = df["datetime_ist"].iloc[0].replace(hour=thresholds.start_hour, minute=0, second=0)
    if start_datetime >= df["datetime_ist"].min() and start_datetime <= df["datetime_ist"].max():
        start_time_ms = int(start_datetime.timestamp() * 1000)
        for row in [1, 2, 3]:
            fig.add_vline(
                x=start_time_ms,
                line_width=2,
                line_dash="dashdot",
                line_color="purple",
                opacity=0.5,
                row=row, col=1
            )
        fig.add_annotation(
            x=start_time_ms,
            y=df["value"].max() + 10,
            text=f"Detection starts ({thresholds.start_hour}:00)",
            showarrow=False,
            font=dict(size=10, color="purple"),
            row=1, col=1
        )

    # Add event markers
    event_colors = {
        "MEAL_START": "#2ECC71",
        "PEAK": "#3498DB"
    }

    y_max = df["value"].max()
    annotation_offset = 0

    # Build a set of event times that are part of merged groups (for lighter rendering)
    merged_event_times = set()
    merged_group_info = {}  # Maps composite event_id to info about the merged group

    if composite_events:
        for comp in composite_events:
            # Check if this composite event has secondary events (i.e., it's a merged group)
            if comp.secondary_events and len(comp.secondary_events) > 0:
                # Store the merged mean time and individual event times
                merged_group_info[comp.event_id] = {
                    "mean_time": comp.start_time,  # This is now the mean time
                    "primary_event": comp.primary_event,
                    "secondary_events": comp.secondary_events,
                    "glucose": comp.glucose_at_start
                }
                # Mark individual event times as merged
                merged_event_times.add(comp.primary_event.detected_at)
                for sec in comp.secondary_events:
                    merged_event_times.add(sec.detected_at)

    # Draw vertical lines for events
    # - PEAK: vertical lines on all plots
    # - MEAL_START: vertical lines only on dG/dt plot (row 2) to show zero-crossings
    for event in events:
        color = event_colors.get(event.event_type, "#333333")
        event_time_ms = int(event.detected_at.timestamp() * 1000)

        if event.event_type == "PEAK":
            # Peak lines on all plots
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=event_time_ms,
                    line_width=2,
                    line_dash="dot",
                    line_color=color,
                    opacity=0.7,
                    row=row, col=1
                )

            fig.add_annotation(
                x=event_time_ms,
                y=event.glucose_at_detection + 10,
                text=f"Peak<br>{event.detected_at.strftime('%H:%M')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                font=dict(size=9, color=color),
                row=1, col=1
            )

        elif event.event_type == "MEAL_START":
            # Meal start lines only on dG/dt plot (row 2) to show zero-crossings
            is_merged = event.detected_at in merged_event_times
            fig.add_vline(
                x=event_time_ms,
                line_width=1 if is_merged else 2,
                line_dash="solid",
                line_color=color,
                opacity=0.4 if is_merged else 0.7,
                row=2, col=1
            )

    # Add merged meal time markers (circle at mean time) - markers only, no vertical lines
    if merged_group_info:
        merged_times = []
        merged_glucose_values = []
        merged_labels = []

        for event_id, info in merged_group_info.items():
            mean_time = info["mean_time"]
            # Get glucose at mean time by interpolating from CGM data
            time_diffs = abs(df["datetime_ist"] - mean_time)
            closest_idx = time_diffs.idxmin()
            glucose_at_mean = df.loc[closest_idx, "value"]

            merged_times.append(mean_time)
            merged_glucose_values.append(glucose_at_mean)
            num_merged = 1 + len(info["secondary_events"])
            merged_labels.append(f"Merged ({num_merged} events)")

        # Add circle markers for merged meal times on glucose plot (no vertical lines)
        fig.add_trace(
            go.Scatter(
                x=merged_times,
                y=merged_glucose_values,
                mode="markers",
                name="Merged Meal Time",
                marker=dict(
                    size=16,
                    color="#27AE60",
                    symbol="circle",
                    line=dict(color="white", width=2)
                ),
                hovertemplate="<b>Merged Meal Time</b><br>Time: %{x|%H:%M}<br>Glucose: %{y:.1f} mg/dL<br>%{customdata}<extra></extra>",
                customdata=merged_labels
            ),
            row=1, col=1
        )
        # Note: Removed vertical lines and annotations for merged meals to reduce clutter

    # Add markers for meal events with classification info in hover (no visible labels)
    if composite_events:
        meal_times = []
        meal_glucose = []
        meal_hover_texts = []
        meal_colors = []
        meal_symbols = []

        for comp in composite_events:
            # Determine classification label and color
            if comp.is_clean:
                label = "✅ Clean"
                color = "#2ECC71"  # Green
                symbol = "circle"
            elif comp.is_no_peak:
                label = "⚠️ No Peak"
                color = "#E67E22"  # Orange
                symbol = "diamond"
            else:
                label = "⚡ Composite"
                color = "#9B59B6"  # Purple
                symbol = "square"

            glucose_at_event = comp.glucose_at_start if comp.glucose_at_start else df["value"].mean()

            meal_times.append(comp.start_time)
            meal_glucose.append(glucose_at_event)
            meal_colors.append(color)
            meal_symbols.append(symbol)

            # Build hover text with all classification details
            hover_parts = [
                f"<b>{comp.event_id}: {label}</b>",
                f"Time: {comp.start_time.strftime('%H:%M')}",
                f"Glucose: {glucose_at_event:.0f} mg/dL"
            ]

            if comp.peak_time:
                hover_parts.append(f"Peak at: {comp.peak_time.strftime('%H:%M')}")
            if comp.time_to_peak_minutes:
                hover_parts.append(f"Time to peak: {comp.time_to_peak_minutes:.0f} min")
            if comp.total_glucose_rise is not None:
                hover_parts.append(f"Glucose rise: +{comp.total_glucose_rise:.0f} mg/dL")
            if comp.secondary_events and len(comp.secondary_events) > 0:
                hover_parts.append(f"Merged events: {1 + len(comp.secondary_events)}")
            if comp.delta_g_to_next_event is not None:
                hover_parts.append(f"ΔG to next: +{comp.delta_g_to_next_event:.0f} mg/dL")

            meal_hover_texts.append("<br>".join(hover_parts))

        # Add markers for meal events (classification visible on hover only)
        fig.add_trace(
            go.Scatter(
                x=meal_times,
                y=meal_glucose,
                mode="markers",
                name="Meal Events",
                marker=dict(
                    size=12,
                    color=meal_colors,
                    symbol="triangle-up",
                    line=dict(color="white", width=1)
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=meal_hover_texts
            ),
            row=1, col=1
        )

    # Add glucose rise annotations from matches
    if matches:
        for match in matches:
            event_time_ms = int(match.event_time.timestamp() * 1000)

            if match.glucose_rise is not None and match.glucose_at_start is not None and match.glucose_at_peak is not None:
                if match.peak_time:
                    peak_time_ms = int(match.peak_time.timestamp() * 1000)

                    # Add annotation showing the rise amount
                    rise_color = "#E74C3C" if match.glucose_rise > 50 else "#F39C12" if match.glucose_rise > 30 else "#27AE60"

                    # Build annotation text
                    annotation_text = f"↑{match.glucose_rise:.0f}"
                    if match.time_to_peak_minutes is not None:
                        annotation_text += f" ({match.time_to_peak_minutes:.0f}m)"

                    fig.add_annotation(
                        x=peak_time_ms,
                        y=match.glucose_at_peak + 5,
                        text=annotation_text,
                        showarrow=False,
                        font=dict(size=10, color=rise_color, weight="bold"),
                        bgcolor="rgba(255,255,255,0.8)",
                        row=1, col=1
                    )

            # For composite events, show delta G to next event if available
            if match.delta_g_to_next_event is not None and not match.is_clean_event:
                # Find approximate position for next event annotation
                fig.add_annotation(
                    x=event_time_ms,
                    y=match.glucose_at_start + (match.delta_g_to_next_event / 2) if match.glucose_at_start else 100,
                    text=f"→+{match.delta_g_to_next_event:.0f}",
                    showarrow=False,
                    font=dict(size=8, color="#9B59B6"),
                    bgcolor="rgba(255,255,255,0.6)",
                    row=1, col=1
                )

    # Plot derived meal times as X markers (from dataset)
    if derived_meal_times is not None and len(derived_meal_times) > 0:
        # Find glucose values at derived meal times by interpolating from CGM data
        derived_times = []
        derived_glucose = []

        for dmt in derived_meal_times:
            if pd.isna(dmt):
                continue
            # Find the closest CGM reading to get approximate glucose value
            time_diffs = abs(df["datetime_ist"] - dmt)
            closest_idx = time_diffs.idxmin()
            derived_times.append(dmt)
            derived_glucose.append(df.loc[closest_idx, "value"])

        if derived_times:
            fig.add_trace(
                go.Scatter(
                    x=derived_times,
                    y=derived_glucose,
                    mode="markers",
                    name="Derived Meal Time (Dataset)",
                    marker=dict(
                        size=14,
                        color="#E74C3C",
                        symbol="x",
                        line=dict(color="#E74C3C", width=3)
                    ),
                    hovertemplate="<b>Derived Meal Time</b><br>Time: %{x|%H:%M}<br>Glucose: %{y:.1f} mg/dL<extra></extra>"
                ),
                row=1, col=1
            )
            # Note: Removed vertical lines for derived meal times to reduce clutter

    # Plot dietary response times as X markers and before/after meal glucose curves
    if dietary_response_df is not None and len(dietary_response_df) > 0:
        from datetime import timedelta

        dietary_times = []
        dietary_glucose = []
        dietary_hover_texts = []

        # Time offsets for before/after meal readings (in minutes)
        # beforemeal_0 = T-60, beforemeal_1 = T-45, beforemeal_2 = T-30, beforemeal_3 = T-15
        # postmeal_0 = T+15, postmeal_1 = T+30, ..., postmeal_7 = T+120
        time_offsets = {
            'beforemeal_0': -60, 'beforemeal_1': -45, 'beforemeal_2': -30, 'beforemeal_3': -15,
            'postmeal_0': 15, 'postmeal_1': 30, 'postmeal_2': 45, 'postmeal_3': 60,
            'postmeal_4': 75, 'postmeal_5': 90, 'postmeal_6': 105, 'postmeal_7': 120
        }

        for _, row in dietary_response_df.iterrows():
            diet_time = row["datetime_ist"]
            if pd.isna(diet_time):
                continue

            # Find the closest CGM reading to get approximate glucose value
            time_diffs = abs(df["datetime_ist"] - diet_time)
            closest_idx = time_diffs.idxmin()
            glucose_val = df.loc[closest_idx, "value"]

            dietary_times.append(diet_time)
            dietary_glucose.append(glucose_val)

            # Build hover text with dietary values
            hover_parts = [
                f"<b>Dietary Response Data</b>",
                f"Time: {diet_time.strftime('%H:%M')}",
                f"Glucose: {glucose_val:.1f} mg/dL",
                f"<b>--- Nutritional Info ---</b>",
                f"Calories: {row.get('Heat', 'N/A'):.1f} kcal" if pd.notna(row.get('Heat')) else "Calories: N/A",
                f"Carbs: {row.get('Carbohydrate', 'N/A'):.1f} g" if pd.notna(row.get('Carbohydrate')) else "Carbs: N/A",
                f"Protein: {row.get('Protein', 'N/A'):.1f} g" if pd.notna(row.get('Protein')) else "Protein: N/A",
                f"Fat: {row.get('Fat', 'N/A'):.1f} g" if pd.notna(row.get('Fat')) else "Fat: N/A",
                f"Fiber: {row.get('DietaryFiber', 'N/A'):.1f} g" if pd.notna(row.get('DietaryFiber')) else "Fiber: N/A",
            ]

            # Add CGM response metrics if available
            if pd.notna(row.get('DeltaMax_G')):
                hover_parts.append(f"<b>--- CGM Response ---</b>")
                hover_parts.append(f"ΔG Max: +{row.get('DeltaMax_G', 0):.1f} mg/dL")
            if pd.notna(row.get('postmeal_max')):
                hover_parts.append(f"Peak Glucose: {row.get('postmeal_max', 0):.1f} mg/dL")
            if pd.notna(row.get('postmeal_maxtime')):
                hover_parts.append(f"Time to Peak: {row.get('postmeal_maxtime', 0):.0f} intervals")

            dietary_hover_texts.append("<br>".join(hover_parts))

            # Plot before/after meal glucose curve for this meal
            curve_times = []
            curve_glucose = []
            curve_hover = []

            for col, offset_min in time_offsets.items():
                if col in row and pd.notna(row[col]):
                    point_time = diet_time + timedelta(minutes=offset_min)
                    curve_times.append(point_time)
                    curve_glucose.append(row[col])

                    # Determine label for hover
                    if offset_min < 0:
                        label = f"T{offset_min} min (before meal)"
                    else:
                        label = f"T+{offset_min} min (after meal)"

                    curve_hover.append(f"<b>{label}</b><br>Time: {point_time.strftime('%H:%M')}<br>Glucose: {row[col]:.1f} mg/dL")

            if curve_times:
                # Add the glucose curve trace for this meal
                # Show legend only for first curve
                show_legend = (len(dietary_times) == 1)  # First meal being processed

                fig.add_trace(
                    go.Scatter(
                        x=curve_times,
                        y=curve_glucose,
                        mode="lines+markers",
                        name="Dataset CGM Window",
                        line=dict(color="#8E44AD", width=2, dash="dash"),
                        marker=dict(size=6, color="#8E44AD", symbol="circle"),
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=curve_hover,
                        legendgroup="dietary_curves",
                        showlegend=show_legend
                    ),
                    row=1, col=1
                )

        if dietary_times:
            # Plot X marker at the meal time (thinner, more visible)
            fig.add_trace(
                go.Scatter(
                    x=dietary_times,
                    y=dietary_glucose,
                    mode="markers",
                    name="Dietary Response (Dataset)",
                    marker=dict(
                        size=14,
                        color="#8E44AD",  # Purple color
                        symbol="x-thin",  # Thinner X marker
                        line=dict(color="#8E44AD", width=2)
                    ),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=dietary_hover_texts
                ),
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        height=850,
        showlegend=True,
        hovermode="x unified",
        title="Simplified Zero-Crossing Meal Detection"
    )

    fig.update_xaxes(tickformat="%H:%M", dtick=3600000 * 2, row=3, col=1)

    # Set dynamic y-axis range for glucose plot (min - 20, max + 20)
    glucose_min = df["value"].min() - 20
    glucose_max = df["value"].max() + 20
    fig.update_yaxes(title_text="Glucose (mg/dL)", range=[glucose_min, glucose_max], row=1, col=1)

    fig.update_yaxes(title_text="dG/dt (mg/dL/min)", row=2, col=1)
    fig.update_yaxes(title_text="d²G/dt² (mg/dL/min²)", row=3, col=1)

    return fig


def create_simple_cgm_plot(cgm_df: pd.DataFrame) -> go.Figure:
    """Create a simple CGM plot without meal detection."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cgm_df["datetime_ist"],
        y=cgm_df["value"],
        mode="lines+markers",
        name="CGM Glucose",
        line=dict(color="#2E86AB", width=2),
        marker=dict(size=4),
        hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<extra></extra>"
    ))

    y_min = max(0, cgm_df["value"].min() - 20)
    y_max = cgm_df["value"].max() + 30

    fig.add_shape(type="line", x0=0, x1=1, y0=70, y1=70,
                  line=dict(color="orange", width=1, dash="dot"),
                  xref="paper", yref="y")
    fig.add_shape(type="line", x0=0, x1=1, y0=140, y1=140,
                  line=dict(color="orange", width=1, dash="dot"),
                  xref="paper", yref="y")
    fig.add_annotation(x=0, y=70, text="Low (70)", showarrow=False,
                       xref="paper", font=dict(size=10, color="orange"), xanchor="right", xshift=-5)
    fig.add_annotation(x=0, y=140, text="High (140)", showarrow=False,
                       xref="paper", font=dict(size=10, color="orange"), xanchor="right", xshift=-5)

    fig.update_layout(
        title="CGM Data - 24 Hour View",
        xaxis_title="Time (IST)",
        yaxis_title="Glucose (mg/dL)",
        hovermode="x unified",
        height=500,
        showlegend=True,
        xaxis=dict(
            tickformat="%H:%M",
            dtick=3600000 * 2,
        ),
        yaxis=dict(
            range=[y_min, y_max]
        )
    )

    return fig
