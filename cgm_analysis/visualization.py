"""
Visualization functions for CGM data and meal detection.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from .models import MealEvent, MealMatch, SimplifiedThresholds
from .derivatives import apply_smoothing, calculate_first_derivative, calculate_second_derivative
from .detection import detect_secondary_meal_events


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

    # Plot passing peaks (green) - peaks that pass dG/dt threshold
    if passing_events:
        passing_times = [e.detected_at for e in passing_events]
        passing_glucose = [e.glucose_at_detection for e in passing_events]
        passing_d2G = [e.d2G_dt2_at_detection for e in passing_events]
        passing_dG = [e.dG_dt_at_detection for e in passing_events]

        fig.add_trace(
            go.Scatter(
                x=passing_times,
                y=passing_glucose,
                mode="markers",
                name="Est. Secondary Meal",
                marker=dict(
                    size=12,
                    color="#27AE60",
                    symbol="diamond",
                    line=dict(color="white", width=1)
                ),
                hovertemplate="<b>Time:</b> %{x|%H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<br><b>dG/dt:</b> %{customdata[0]:.4f}<br><b>d²G/dt²:</b> %{customdata[1]:.4f}<extra>Est. Secondary Meal</extra>",
                customdata=list(zip(passing_dG, passing_d2G))
            ),
            row=1, col=1
        )

        for event in passing_events:
            sec_meal_time = event.detected_at
            sec_meal_time_ms = int(sec_meal_time.timestamp() * 1000)
            sec_meal_glucose = event.glucose_at_detection

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

        # Add markers on d²G/dt² plot for passing peaks
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

    # Add glucose rise annotations from matches with clean/composite classification
    if matches:
        for match in matches:
            event_time_ms = int(match.event_time.timestamp() * 1000)

            # Determine event classification styling
            if match.is_clean_event:
                event_label = "Clean"
                event_border_color = "#2ECC71"  # Green for clean
            else:
                event_label = "Composite"
                event_border_color = "#9B59B6"  # Purple for composite

            if match.glucose_rise is not None and match.glucose_at_start is not None and match.glucose_at_peak is not None:
                # Add shaded region between start and peak
                if match.peak_time:
                    peak_time_ms = int(match.peak_time.timestamp() * 1000)

                    # Add annotation showing the rise amount and classification
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

                    # Add classification label at event start
                    fig.add_annotation(
                        x=event_time_ms,
                        y=match.glucose_at_start - 8,
                        text=event_label,
                        showarrow=False,
                        font=dict(size=8, color=event_border_color),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor=event_border_color,
                        borderwidth=1,
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
