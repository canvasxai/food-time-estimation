"""
Visualization functions for CGM data and meal detection.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from .models import MealEvent, MealMatch, SimplifiedThresholds
from .derivatives import apply_smoothing, calculate_first_derivative, calculate_second_derivative


def create_simplified_derivative_plot(cgm_df: pd.DataFrame, events: List[MealEvent],
                                       thresholds: SimplifiedThresholds,
                                       matches: List[MealMatch] = None) -> go.Figure:
    """
    Create a plot for the simplified detection method showing:
    1. Raw and smoothed glucose curves
    2. First derivative (dG/dt) with zero line and threshold bands
    3. Second derivative (d²G/dt²)
    4. Detected events marked
    5. Matched meal events annotated (if matches provided)
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

    # Find and plot second derivative maxima on glucose plot
    d2G_dt2_values = df["d2G_dt2"].values
    dG_dt_values = df["dG_dt"].values

    # Determine primary date for filtering
    primary_date = df["datetime_ist"].dt.date.mode().iloc[0]

    maxima_indices = []
    for i in range(1, len(d2G_dt2_values) - 1):
        # A meal event requires:
        # 1. d²G/dt² is at a local maximum (acceleration peak)
        # 2. d²G/dt² > 0 (positive acceleration)
        # 3. dG/dt > 0 (glucose is increasing)
        # When dG/dt <= 0 (glucose falling), it's a "change event", not a meal event
        if (d2G_dt2_values[i] > d2G_dt2_values[i - 1] and
            d2G_dt2_values[i] > d2G_dt2_values[i + 1] and
            d2G_dt2_values[i] > 0 and  # Positive acceleration (d²G/dt² > 0)
            dG_dt_values[i] > 0):  # Glucose must be rising (dG/dt > 0)
            detection_datetime = df.iloc[i]["datetime_ist"]
            detection_date = detection_datetime.date()
            detection_hour = detection_datetime.hour

            if detection_date == primary_date and detection_hour < thresholds.start_hour:
                continue

            event_idx = i - 1
            maxima_indices.append(event_idx)

    if maxima_indices:
        maxima_times = df.iloc[maxima_indices]["datetime_ist"]
        maxima_glucose = df.iloc[maxima_indices]["smoothed"]
        maxima_d2G = df.iloc[maxima_indices]["d2G_dt2"]

        fig.add_trace(
            go.Scatter(
                x=maxima_times,
                y=maxima_glucose,
                mode="markers",
                name="Est. Secondary Meal",
                marker=dict(
                    size=12,
                    color="#27AE60",
                    symbol="diamond",
                    line=dict(color="white", width=1)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<br><b>d²G/dt²:</b> %{customdata:.4f}<extra>Est. Secondary Meal</extra>",
                customdata=maxima_d2G
            ),
            row=1, col=1
        )

        for idx in maxima_indices:
            sec_meal_time = df.iloc[idx]["datetime_ist"]
            sec_meal_time_ms = int(sec_meal_time.timestamp() * 1000)
            sec_meal_glucose = df.iloc[idx]["smoothed"]

            for row in [1, 2, 3]:
                fig.add_vline(
                    x=sec_meal_time_ms,
                    line_width=1.5,
                    line_dash="dash",
                    line_color="#27AE60",
                    opacity=0.6,
                    row=row, col=1
                )

            fig.add_annotation(
                x=sec_meal_time_ms,
                y=sec_meal_glucose + 8,
                text=f"Est. Sec. Meal<br>{sec_meal_time.strftime('%H:%M')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#27AE60",
                font=dict(size=9, color="#27AE60"),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(
                x=maxima_times,
                y=maxima_d2G,
                mode="markers",
                name="Est. Secondary Meal",
                marker=dict(
                    size=10,
                    color="#27AE60",
                    symbol="diamond",
                    line=dict(color="white", width=1)
                ),
                showlegend=False,
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

    for event in events:
        color = event_colors.get(event.event_type, "#333333")
        event_time_ms = int(event.detected_at.timestamp() * 1000)

        for row in [1, 2, 3]:
            fig.add_vline(
                x=event_time_ms,
                line_width=2,
                line_dash="solid" if event.event_type == "MEAL_START" else "dot",
                line_color=color,
                opacity=0.7,
                row=row, col=1
            )

        if event.event_type == "MEAL_START" and event.estimated_meal_time:
            est_meal_time_ms = int(event.estimated_meal_time.timestamp() * 1000)

            for row in [1, 2, 3]:
                fig.add_vline(
                    x=est_meal_time_ms,
                    line_width=2,
                    line_dash="solid",
                    line_color="#27AE60",
                    opacity=0.8,
                    row=row, col=1
                )

            fig.add_annotation(
                x=est_meal_time_ms,
                y=y_max + 15 + annotation_offset,
                text=f"Est. Meal<br>{event.estimated_meal_time.strftime('%H:%M')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#27AE60",
                font=dict(size=10, color="#27AE60", weight="bold"),
                row=1, col=1
            )
            annotation_offset += 25

        elif event.event_type == "PEAK":
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

    # Add matched meal annotations
    if matches:
        for match in matches:
            event_time = match.event_time
            event_time_ms = int(event_time.timestamp() * 1000)
            meal_time_ms = int(match.meal_time.timestamp() * 1000)

            time_diffs = abs(df["datetime_ist"] - event_time)
            closest_idx = time_diffs.idxmin()
            glucose_at_event = df.loc[closest_idx, "smoothed"]

            if match.composite_score >= 0.85:
                match_color = "#2ECC71"
            elif match.composite_score >= 0.75:
                match_color = "#F39C12"
            else:
                match_color = "#E74C3C"

            if match.event_type == "MEAL_START":
                event_type_display = "Est. Meal"
            elif match.event_type == "SECONDARY_MEAL":
                event_type_display = "Est. Sec. Meal"
            else:
                event_type_display = match.event_type

            fig.add_trace(
                go.Scatter(
                    x=[event_time],
                    y=[glucose_at_event],
                    mode="markers",
                    name=f"Matched: {match.meal_slot}",
                    marker=dict(
                        size=16,
                        color=match_color,
                        symbol="star",
                        line=dict(color="white", width=2)
                    ),
                    hovertemplate=(
                        f"<b>MATCHED EVENT</b><br>"
                        f"<b>Meal Slot:</b> {match.meal_slot}<br>"
                        f"<b>Meal Name:</b> {match.meal_name[:25]}{'...' if len(match.meal_name) > 25 else ''}<br>"
                        f"<b>Event Time:</b> {match.event_time.strftime('%H:%M')}<br>"
                        f"<b>Event Type:</b> {event_type_display}<br>"
                        f"<b>---Scoring---</b><br>"
                        f"<b>Carbs:</b> {match.carbs:.1f}g<br>"
                        f"<b>Carbs/Rise Score:</b> {match.s_size:.2f}<br>"
                        f"<b>Match Score:</b> {match.composite_score:.1%}<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.add_annotation(
                x=event_time_ms,
                y=glucose_at_event - 15,
                text=f"{match.meal_slot}<br>{match.composite_score:.0%}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=match_color,
                font=dict(size=9, color=match_color, weight="bold"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=match_color,
                borderwidth=1,
                row=1, col=1
            )

            meal_time_diffs = abs(df["datetime_ist"] - match.meal_time)
            meal_closest_idx = meal_time_diffs.idxmin()
            glucose_at_meal = df.loc[meal_closest_idx, "smoothed"]

            fig.add_trace(
                go.Scatter(
                    x=[match.meal_time, event_time],
                    y=[glucose_at_meal, glucose_at_event],
                    mode="lines",
                    line=dict(color=match_color, width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip"
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=[match.meal_time],
                    y=[glucose_at_meal],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=match_color,
                        symbol="circle",
                        line=dict(color="white", width=1)
                    ),
                    hoverinfo="skip",
                    showlegend=False
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
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1)
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
