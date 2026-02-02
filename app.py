"""
CGM Visualization and Meal Detection App

A Streamlit application for visualizing CGM data and detecting meal times using derivative analysis.
"""

import streamlit as st
import pandas as pd
import json

from cgm_analysis import (
    # Data loading
    get_available_users,
    load_cgm_data,
    load_meals_data,
    load_dietary_response_data,
    get_available_dates,
    filter_data_for_date,
    USER_DATA_PATH,
    # Models
    SimplifiedThresholds,
    # Detection
    detect_meal_events_simplified,
    # Matching
    match_meals_to_events,
    # Visualization
    create_simplified_derivative_plot,
    create_simple_cgm_plot,
)


def main():
    st.set_page_config(
        page_title="CGM Meal Detection",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 CGM Visualization & Meal Detection")
    st.markdown("Visualize CGM data and detect meal times using derivative analysis.")

    # Sidebar for controls
    st.sidebar.header("Controls")

    # User selection
    users = get_available_users()
    selected_user = st.sidebar.selectbox("Select User", users)

    # Load data for selected user
    cgm_df = load_cgm_data(selected_user)
    meals_df = load_meals_data(selected_user)
    dietary_response_df = load_dietary_response_data(selected_user)

    # Date selection
    available_dates = get_available_dates(cgm_df)
    # Get dates that have logged meals
    dates_with_meals = set(meals_df["date"].unique()) if len(meals_df) > 0 else set()

    def format_date_with_meal_indicator(date):
        date_str = date.strftime("%Y-%m-%d (%A)")
        if date in dates_with_meals:
            return f"🍽️ {date_str}"
        return f"    {date_str}"

    selected_date = st.sidebar.selectbox(
        "Select Date",
        available_dates,
        format_func=format_date_with_meal_indicator
    )

    # Extend view past midnight
    extend_hours = st.sidebar.slider(
        "Extend past midnight (hours)",
        min_value=0, max_value=6, value=2,
        help="Show CGM data extending into the next day to see dinner effects"
    )

    # Filter data for selected date
    day_cgm = filter_data_for_date(cgm_df, selected_date, extend_hours=extend_hours)
    day_meals = filter_data_for_date(meals_df, selected_date, extend_hours=extend_hours)
    day_dietary = filter_data_for_date(dietary_response_df, selected_date, extend_hours=extend_hours) if len(dietary_response_df) > 0 else pd.DataFrame()

    # Detection method selection
    st.sidebar.header("Meal Detection")
    enable_detection = st.sidebar.checkbox("Enable Meal Detection", value=True)

    # Main content area
    if enable_detection:
        # Simplified zero-crossing method
        st.sidebar.header("Detection Parameters")

        smoothing_window = st.sidebar.slider(
            "Smoothing Window",
            min_value=3, max_value=9, value=5, step=2,
            help="Rolling average window size (must be odd)"
        )

        min_derivative = st.sidebar.slider(
            "Min Derivative Magnitude",
            min_value=0.01, max_value=1.0, value=0.01, step=0.01,
            help="Minimum |dG/dt| to consider a zero-crossing significant. Higher = fewer events (filters noise)."
        )

        start_hour = st.sidebar.slider(
            "Start Hour",
            min_value=5, max_value=12, value=7, step=1,
            help="Only detect events after this hour (24h format)"
        )

        end_hour = st.sidebar.slider(
            "End Hour",
            min_value=20, max_value=24, value=24, step=1,
            help="Stop detecting events after this hour (24h format, 24 = midnight)"
        )

        absorption_lag = st.sidebar.slider(
            "Absorption Lag (min)",
            min_value=0, max_value=20, value=0, step=1,
            help="Time from eating to detectable glucose rise"
        )

        secondary_dg_dt_threshold = st.sidebar.slider(
            "Secondary Meal dG/dt Threshold",
            min_value=-0.5, max_value=0.5, value=-0.1, step=0.01,
            help="Min dG/dt for secondary meal detection. Negative values allow detection when glucose is slightly declining."
        )

        event_merge_gap = st.sidebar.slider(
            "Event Merge Gap (min)",
            min_value=0, max_value=60, value=30, step=5,
            help="Merge meal events within this many minutes. Set to 0 to disable merging."
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Peak Detection (scipy)**")

        peak_prominence = st.sidebar.slider(
            "Peak Prominence (mg/dL)",
            min_value=1.0, max_value=30.0, value=5.0, step=1.0,
            help="Minimum prominence for peak detection. Higher values detect only more prominent peaks."
        )

        peak_distance = st.sidebar.slider(
            "Peak Distance (samples)",
            min_value=1, max_value=12, value=1, step=1,
            help="Minimum samples between peaks. At 15-min intervals: 1=15min, 4=1hr."
        )

        simplified_thresholds = SimplifiedThresholds(
            smoothing_window=smoothing_window,
            min_derivative_magnitude=min_derivative,
            start_hour=start_hour,
            end_hour=end_hour,
            meal_absorption_lag=absorption_lag,
            secondary_meal_dg_dt_threshold=secondary_dg_dt_threshold,
            event_merge_gap_minutes=event_merge_gap,
            peak_prominence=peak_prominence,
            peak_distance=peak_distance
        )

        if len(day_cgm) > 0:
            # Detect events using simplified method
            events = detect_meal_events_simplified(day_cgm, simplified_thresholds)

            # Step 1: Detect and classify composite events (based on CGM only - no meals needed)
            # This is the single source of truth for classification (clean/composite/no_peak)
            composite_events_for_plot = []
            discarded_events_for_plot = []
            if events:
                from cgm_analysis.detection import detect_composite_events
                composite_events_for_plot, discarded_events_for_plot = detect_composite_events(
                    events,
                    cgm_df=day_cgm,
                    merge_gap_minutes=simplified_thresholds.event_merge_gap_minutes,
                    max_peak_window_minutes=180,
                    return_discarded=True
                )

            # Step 2: Match meals to events (only when meals exist - this is optional)
            matches_for_plot = []
            validation_results_for_plot = {}
            if len(day_meals) > 0 and events:
                matches_for_plot, _, _, validation_results_for_plot = match_meals_to_events(
                    day_meals, events, cgm_df=day_cgm, thresholds=simplified_thresholds
                )

            # Get derived meal times if available
            derived_meal_times = None
            if "derived_meal_time_ist" in day_meals.columns:
                derived_meal_times = day_meals["derived_meal_time_ist"].tolist()

            # Create and show plot (with matches, composite events, and discarded events)
            fig = create_simplified_derivative_plot(
                day_cgm, events, simplified_thresholds,
                matches=matches_for_plot,
                derived_meal_times=derived_meal_times,
                composite_events=composite_events_for_plot,
                discarded_events=discarded_events_for_plot,
                dietary_response_df=day_dietary if len(day_dietary) > 0 else None
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show detected events
            if events:
                meal_events = [e for e in events if e.event_type == "MEAL_START"]
                peak_events = [e for e in events if e.event_type == "PEAK"]

                if meal_events:
                    st.subheader("🍽️ Estimated Meal Times")
                    st.markdown("*Detected when dG/dt crosses from negative to positive (local minimum):*")

                    meal_cols = st.columns(min(len(meal_events), 4))
                    for i, event in enumerate(meal_events):
                        with meal_cols[i % 4]:
                            hour = event.estimated_meal_time.hour
                            if 5 <= hour < 11:
                                meal_label = "Breakfast"
                            elif 11 <= hour < 15:
                                meal_label = "Lunch"
                            elif 15 <= hour < 18:
                                meal_label = "Snack"
                            else:
                                meal_label = "Dinner"

                            st.metric(
                                label=f"{meal_label}",
                                value=event.estimated_meal_time.strftime("%H:%M"),
                                delta=f"{event.confidence:.0%} confidence"
                            )

                if peak_events:
                    st.subheader("📈 Post-Prandial Peaks")
                    st.markdown("*Detected when dG/dt crosses from positive to negative (local maximum):*")

                    peak_cols = st.columns(min(len(peak_events), 4))
                    for i, event in enumerate(peak_events):
                        with peak_cols[i % 4]:
                            st.metric(
                                label=f"Peak @ {event.glucose_at_detection:.0f} mg/dL",
                                value=event.detected_at.strftime("%H:%M"),
                                delta=f"{event.confidence:.0%} confidence"
                            )

                # Show all events in detailed table (chronological with classification)
                with st.expander(f"📊 All Detected Events ({len(events)} total)", expanded=True):
                    # Build a unified chronological view using composite events
                    events_data = []

                    # Create a mapping from event to its composite classification
                    event_to_composite = {}
                    if composite_events_for_plot:
                        for comp in composite_events_for_plot:
                            # Map primary event
                            event_to_composite[comp.primary_event.detected_at] = comp
                            # Map secondary events
                            for sec in comp.secondary_events:
                                event_to_composite[sec.detected_at] = comp

                    # Sort all events by time
                    sorted_events = sorted(events, key=lambda e: e.detected_at)

                    for event in sorted_events:
                        comp = event_to_composite.get(event.detected_at)

                        # Determine classification
                        if event.event_type == "PEAK":
                            classification = "—"
                            event_id = "—"
                            merged_count = "—"
                            associated_peak = "—"
                            glucose_rise = "—"
                            time_to_peak = "—"
                        else:
                            # Meal event - get classification from composite
                            if comp:
                                if comp.is_clean:
                                    classification = "✅ Clean"
                                elif comp.is_no_peak:
                                    classification = "⚠️ No Peak"
                                else:
                                    classification = "⚡ Composite"

                                event_id = comp.event_id

                                # Count merged events (primary + secondary in same composite)
                                merged_count = 1 + len(comp.secondary_events)
                                if merged_count > 1:
                                    merged_count = f"🔗 {merged_count}"
                                else:
                                    merged_count = "1"

                                # Associated peak
                                if comp.peak_time:
                                    associated_peak = comp.peak_time.strftime("%H:%M")
                                else:
                                    associated_peak = "—"

                                # Glucose rise
                                if comp.total_glucose_rise is not None:
                                    glucose_rise = f"+{comp.total_glucose_rise:.0f}"
                                else:
                                    glucose_rise = "—"

                                # Time to peak
                                if comp.time_to_peak_minutes is not None:
                                    time_to_peak = f"{comp.time_to_peak_minutes:.0f} min"
                                else:
                                    time_to_peak = "—"
                            else:
                                # Event not in composite_events - it was discarded (insufficient ΔG)
                                classification = "🚫 Discarded"
                                event_id = "—"
                                merged_count = "—"
                                associated_peak = "—"
                                glucose_rise = "—"
                                time_to_peak = "—"

                        # Event type display
                        if event.event_type == "MEAL_START":
                            type_display = "🟢 MEAL_START"
                        elif event.event_type == "SECONDARY_MEAL":
                            type_display = "🟡 SECONDARY"
                        elif event.event_type == "PEAK":
                            type_display = "🔵 PEAK"
                        else:
                            type_display = event.event_type

                        row = {
                            "Time": event.detected_at.strftime("%H:%M"),
                            "Type": type_display,
                            "Classification": classification,
                            "Event ID": event_id,
                            "Merged": merged_count,
                            "Peak At": associated_peak,
                            "T→Peak": time_to_peak,
                            "ΔG": glucose_rise,
                            "Glucose": f"{event.glucose_at_detection:.0f}",
                            "dG/dt": f"{event.dG_dt_at_detection:.2f}"
                        }
                        events_data.append(row)

                    events_df = pd.DataFrame(events_data)
                    st.dataframe(
                        events_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Time": st.column_config.TextColumn("Time", help="Time of event detection"),
                            "Type": st.column_config.TextColumn("Type", help="MEAL_START, SECONDARY, or PEAK"),
                            "Classification": st.column_config.TextColumn("Class", help="Clean ✅, Composite ⚡, or No Peak ⚠️"),
                            "Event ID": st.column_config.TextColumn("Event", help="Composite event ID"),
                            "Merged": st.column_config.TextColumn("Merged", help="Number of events merged together"),
                            "Peak At": st.column_config.TextColumn("Peak At", help="Time of associated peak"),
                            "T→Peak": st.column_config.TextColumn("T→Peak", help="Time from event to peak"),
                            "ΔG": st.column_config.TextColumn("ΔG", help="Total glucose rise to peak (mg/dL)"),
                            "Glucose": st.column_config.TextColumn("Glucose", help="Glucose at detection (mg/dL)"),
                            "dG/dt": st.column_config.TextColumn("dG/dt", help="Rate of glucose change")
                        }
                    )

                    st.markdown(f"""
                    **Event Types:**
                    - 🟢 **MEAL_START**: dG/dt crosses from negative to positive (local minimum = meal onset)
                    - 🟡 **SECONDARY**: Additional meal detected during glucose rise (d²G/dt² maximum)
                    - 🔵 **PEAK**: Local maximum in glucose (scipy peak detection)

                    **Classifications:**
                    - ✅ **Clean**: Single meal → peak (clear glucose response)
                    - ⚡ **Composite**: Multiple meals share the same peak (stacked meals)
                    - ⚠️ **No Peak**: Meal detected but no peak found within 180 min window
                    - 🚫 **Discarded**: Filtered out due to insufficient ΔG (< 5 mg/dL) to next event (noise)

                    **Merged Events (🔗):** Events within {event_merge_gap} min are merged as one meal occasion.

                    **Chart Legend:**
                    - 🟢 **Solid green lines (dG/dt plot)**: Zero-crossings where meal events detected
                    - 🔵 **Dotted blue lines**: Peak times (post-prandial peak)
                    - 🟣 **Dash-dot purple line**: Detection start time ({start_hour}:00)
                    - ⬜ **Gray shaded zone**: Noise threshold (crossings here are ignored)
                    """)

                # Meal matching analysis section
                if len(day_meals) > 0:
                    st.divider()
                    st.subheader("🔗 Meal-to-CGM Event Matching Analysis")
                    st.markdown("*Matching logged meals to detected CGM events using composite event detection.*")

                    matches, unmatched_indices, composite_events, validation_results = match_meals_to_events(
                        day_meals, events, cgm_df=day_cgm, thresholds=simplified_thresholds, debug=True
                    )

                    # Show composite events summary
                    if composite_events:
                        clean_events = sum(1 for c in composite_events if c.is_clean)
                        composite_count = sum(1 for c in composite_events if not c.is_clean and not c.is_no_peak)
                        no_peak_count = sum(1 for c in composite_events if c.is_no_peak)

                        summary_parts = [f"{clean_events} clean ✓", f"{composite_count} composite ⚡"]
                        if no_peak_count > 0:
                            summary_parts.append(f"{no_peak_count} no peak ⚠️")

                        st.info(f"📦 Detected **{len(composite_events)} events**: {', '.join(summary_parts)}")

                        with st.expander(f"📦 Event Classification Detail ({len(composite_events)} events)", expanded=False):
                            for comp in composite_events:
                                # Determine classification label and icon
                                if comp.is_clean:
                                    label = "✅ Clean"
                                elif comp.is_no_peak:
                                    label = "⚠️ No Peak"
                                else:
                                    label = "⚡ Composite"

                                st.markdown(f"**{comp.event_id}** ({label})")
                                st.markdown(f"- Time: {comp.start_time.strftime('%H:%M')}")

                                # Show merged events count if any
                                if comp.secondary_events and len(comp.secondary_events) > 0:
                                    st.markdown(f"- Merged: {1 + len(comp.secondary_events)} events")

                                if comp.is_clean:
                                    if comp.time_to_peak_minutes:
                                        st.markdown(f"- Time to peak: {comp.time_to_peak_minutes:.0f} min")
                                    if comp.total_glucose_rise and comp.glucose_at_start is not None and comp.glucose_at_peak is not None:
                                        st.markdown(f"- Glucose rise: **{comp.total_glucose_rise:.0f} mg/dL** ({comp.glucose_at_start:.0f} → {comp.glucose_at_peak:.0f})")

                                elif comp.is_no_peak:
                                    st.markdown(f"- No peak detected within 180 min window")
                                    if comp.glucose_at_start is not None:
                                        st.markdown(f"- Glucose at start: {comp.glucose_at_start:.0f} mg/dL")

                                else:  # Composite
                                    if comp.time_to_peak_minutes:
                                        st.markdown(f"- Time to peak: {comp.time_to_peak_minutes:.0f} min")
                                    if comp.total_glucose_rise and comp.glucose_at_start is not None and comp.glucose_at_peak is not None:
                                        st.markdown(f"- Total glucose rise: **{comp.total_glucose_rise:.0f} mg/dL** ({comp.glucose_at_start:.0f} → {comp.glucose_at_peak:.0f})")
                                    if comp.delta_g_to_next_event is not None and comp.next_event_time:
                                        st.markdown(f"- ΔG to next event: **{comp.delta_g_to_next_event:.0f} mg/dL** (at {comp.next_event_time.strftime('%H:%M')})")

                                    if comp.event_id in validation_results:
                                        val = validation_results[comp.event_id]
                                        if val["actual_rise"] is not None:
                                            status = "✅" if val["is_valid"] else "⚠️"
                                            st.markdown(f"- Validation: {status} Expected rise: {val['expected_rise']:.0f} mg/dL, Actual: {val['actual_rise']:.0f} mg/dL (confidence: {val['confidence']:.0%})")

                                st.markdown("---")

                    if matches:
                        matches_sorted = sorted(matches, key=lambda m: m.meal_time)

                        match_data = []
                        for match in matches_sorted:
                            if match.event_type == "MEAL_START":
                                event_type_display = "Est. Meal"
                            elif match.event_type == "SECONDARY_MEAL":
                                event_type_display = "Est. Secondary"
                            else:
                                event_type_display = match.event_type

                            # Show classification (Clean ✓ or Composite ⚡)
                            if match.is_clean_event:
                                classification = "Clean ✓"
                            else:
                                classification = "Composite ⚡"

                            # Format glucose rise
                            if match.glucose_rise is not None:
                                glucose_rise_str = f"{match.glucose_rise:.0f}"
                            else:
                                glucose_rise_str = "N/A"

                            # Format time to peak
                            if match.time_to_peak_minutes is not None:
                                time_to_peak_str = f"{match.time_to_peak_minutes:.0f}"
                            else:
                                time_to_peak_str = "N/A"

                            # Format delta G to next event (for composite)
                            if match.delta_g_to_next_event is not None:
                                delta_next_str = f"+{match.delta_g_to_next_event:.0f}"
                            else:
                                delta_next_str = "-"

                            match_data.append({
                                "Meal Slot": match.meal_slot,
                                "Meal Name": match.meal_name[:30] + "..." if len(match.meal_name) > 30 else match.meal_name,
                                "Logged Time": match.meal_time.strftime("%H:%M"),
                                "Event Type": event_type_display,
                                "Classification": classification,
                                "CGM Time": match.event_time.strftime("%H:%M"),
                                "Time Offset": f"{match.time_offset_minutes:+.0f} min",
                                "ΔG Total": glucose_rise_str,
                                "ΔG Next": delta_next_str,
                                "T→Peak": time_to_peak_str,
                                "Carbs": f"{match.carbs:.0f}g",
                                "Score": f"{match.composite_score:.1%}",
                            })

                        match_df = pd.DataFrame(match_data)

                        st.dataframe(
                            match_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Classification": st.column_config.TextColumn("Type", help="Clean ✓ = clear single meal response, Composite ⚡ = stacked meals"),
                                "ΔG Total": st.column_config.TextColumn("ΔG Total", help="Total glucose rise from start to peak (mg/dL)"),
                                "ΔG Next": st.column_config.TextColumn("ΔG Next", help="Glucose rise to next event (for composite events)"),
                                "T→Peak": st.column_config.TextColumn("T→Peak", help="Time from event to peak (minutes)"),
                                "Carbs": st.column_config.TextColumn("Carbs", help="Carbohydrates in the meal"),
                                "Score": st.column_config.TextColumn("Score", help="Composite matching score"),
                                "Event Type": st.column_config.TextColumn("Event Type", help="Type of CGM event matched")
                            }
                        )

                        # Summary statistics
                        avg_composite = sum(m.composite_score for m in matches) / len(matches)
                        avg_offset = sum(m.time_offset_minutes for m in matches) / len(matches)
                        clean_matches = sum(1 for m in matches if m.is_clean_event)
                        composite_matches = len(matches) - clean_matches

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Match Score", f"{avg_composite:.1%}")
                        with col2:
                            st.metric("Avg Time Offset", f"{avg_offset:+.0f} min")
                        with col3:
                            st.metric("Meals Matched", f"{len(matches)}/{len(day_meals)}")
                        with col4:
                            st.metric("Clean / Composite", f"{clean_matches} / {composite_matches}")

                        with st.expander("📐 Scoring Formula Details"):
                            st.markdown("""
                            **Composite Event Detection:**

                            The algorithm first groups events into "composite events":
                            - **Clean Event**: MEAL_START → PEAK (single meal, clear glucose response)
                            - **Stacked Event**: MEAL_START → SECONDARY_MEAL(s) → PEAK (multiple meals eaten close together)

                            **Scoring for Clean Events:**
                            ```
                            S_composite = 0.2·S_time + 0.4·S_slot + 0.2·S_physio + 0.2·S_size
                            ```

                            **Scoring for Stacked Events (⚡):**

                            For stacked meals, we can't reliably calculate:
                            - **S_physio**: Time-to-peak is convoluted by multiple meals
                            - **S_size**: Can't attribute glucose rise to individual meals

                            So we use a simplified formula:
                            ```
                            S_composite = 0.4·S_time + 0.6·S_slot
                            ```

                            Multiple meals can match to the same composite event.
                            The combined glycemic load is validated against the total glucose rise.

                            **Validation for Stacked Events:**
                            - Combined GL proxy of matched meals is calculated
                            - Expected glucose rise is estimated
                            - Compared against actual glucose rise
                            - ✅ = within 30% tolerance, ⚠️ = exceeds tolerance
                            """)

                    # Unmatched events section
                    if unmatched_indices:
                        st.divider()
                        st.subheader("❓ Unmatched CGM Events")
                        st.markdown("*These events didn't match any logged meal - possible unlogged meals or noise.*")

                        unmatched_data = []
                        for idx in unmatched_indices:
                            comp = composite_events[idx]
                            classification = "Clean ✓" if comp.is_clean else "Composite ⚡"
                            secondary_count = len(comp.secondary_events)

                            # Format time to peak
                            time_to_peak_str = f"{comp.time_to_peak_minutes:.0f} min" if comp.time_to_peak_minutes else "N/A"

                            # Format delta G to next event
                            delta_next_str = f"+{comp.delta_g_to_next_event:.0f}" if comp.delta_g_to_next_event else "-"

                            unmatched_data.append({
                                "Event ID": comp.event_id,
                                "Classification": classification,
                                "Start Time": comp.start_time.strftime("%H:%M"),
                                "Peak Time": comp.peak_time.strftime("%H:%M") if comp.peak_time else "N/A",
                                "T→Peak": time_to_peak_str,
                                "Secondary Events": secondary_count,
                                "ΔG Total": f"{comp.total_glucose_rise:.0f}" if comp.total_glucose_rise else "N/A",
                                "ΔG Next": delta_next_str,
                                "Start Glucose": f"{comp.glucose_at_start:.0f}"
                            })

                        unmatched_df = pd.DataFrame(unmatched_data)
                        st.dataframe(unmatched_df, use_container_width=True, hide_index=True)

                        st.info("💡 These events may represent unlogged snacks or meals. Consider reviewing your meal log for this day.")

            else:
                st.info("No meal events detected. Try lowering the 'Min Derivative Magnitude' to be more sensitive.")
        else:
            st.warning(f"No CGM data available for {selected_date}")

    else:
        # No detection - just show CGM plot
        col1, col2 = st.columns([3, 1])

        with col1:
            if len(day_cgm) > 0:
                fig = create_simple_cgm_plot(day_cgm)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No CGM data available for {selected_date}")

        with col2:
            st.subheader("📈 Day Summary")
            if len(day_cgm) > 0:
                st.metric("Average Glucose", f"{day_cgm['value'].mean():.1f} mg/dL")
                st.metric("Min", f"{day_cgm['value'].min():.1f} mg/dL")
                st.metric("Max", f"{day_cgm['value'].max():.1f} mg/dL")
                st.metric("Readings", len(day_cgm))

    # Meals table
    st.subheader("🍽️ Logged Meals for Selected Date")

    if len(day_meals) > 0:
        display_meals = day_meals[["time", "meal_slot", "meal_name", "calories",
                                    "protein", "carbohydrates", "fat", "fibre"]].copy()
        display_meals.columns = ["Time (IST)", "Meal Slot", "Meal Name", "Calories",
                                 "Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)"]

        for col in ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)"]:
            display_meals[col] = display_meals[col].round(1)

        st.dataframe(display_meals, use_container_width=True, hide_index=True)
    else:
        st.info(f"No meals logged for {selected_date}")

    # User info in expander
    with st.expander("📋 User Information"):
        static_path = USER_DATA_PATH / selected_user / "static_data.json"
        with open(static_path) as f:
            user_info = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Age:** {user_info.get('age', 'N/A')}")
            st.write(f"**Gender:** {user_info.get('gender', 'N/A')}")
        with col2:
            st.write(f"**Weight:** {user_info.get('weight', 'N/A')} kg")
            st.write(f"**Diabetes Type:** {user_info.get('diabetes_type', 'N/A')}")
        with col3:
            st.write(f"**Cuisine Preference:** {', '.join(user_info.get('cuisine_pref', []))}")


if __name__ == "__main__":
    main()
