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

    # Date selection
    available_dates = get_available_dates(cgm_df)
    selected_date = st.sidebar.selectbox(
        "Select Date",
        available_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d (%A)")
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
            min_value=5, max_value=10, value=7, step=1,
            help="Only detect events after this hour (24h format)"
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
            meal_absorption_lag=absorption_lag,
            secondary_meal_dg_dt_threshold=secondary_dg_dt_threshold,
            event_merge_gap_minutes=event_merge_gap,
            peak_prominence=peak_prominence,
            peak_distance=peak_distance
        )

        if len(day_cgm) > 0:
            # Detect events using simplified method
            events = detect_meal_events_simplified(day_cgm, simplified_thresholds)

            # Perform matching first so we can show matches on the plot
            matches_for_plot = []
            composite_events_for_plot = []
            validation_results_for_plot = {}
            if len(day_meals) > 0 and events:
                matches_for_plot, _, composite_events_for_plot, validation_results_for_plot = match_meals_to_events(
                    day_meals, events, cgm_df=day_cgm, thresholds=simplified_thresholds
                )

            # Create and show plot (with matches if available)
            fig = create_simplified_derivative_plot(day_cgm, events, simplified_thresholds, matches=matches_for_plot)
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

                # Show all events in detailed table
                with st.expander(f"📊 All Detected Events ({len(events)} total)", expanded=False):
                    events_data = []
                    for event in events:
                        row = {
                            "Type": event.event_type,
                            "Detected At": event.detected_at.strftime("%H:%M:%S"),
                            "Est. Meal Time": event.estimated_meal_time.strftime("%H:%M:%S") if event.estimated_meal_time else "N/A",
                            "Confidence": f"{event.confidence:.0%}",
                            "dG/dt": f"{event.dG_dt_at_detection:.3f}",
                            "Glucose": f"{event.glucose_at_detection:.1f}"
                        }
                        events_data.append(row)

                    events_df = pd.DataFrame(events_data)
                    st.dataframe(events_df, use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    **Event Types:**
                    - 🟢 **MEAL_START**: dG/dt crosses from negative to positive (local minimum = meal onset)
                    - 🔵 **PEAK**: dG/dt crosses from positive to negative (local maximum = post-prandial peak)

                    **Chart Legend:**
                    - 🟢 **Solid green lines**: Estimated meal times (when you likely ate)
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
                        composite_count = len(composite_events) - clean_events

                        st.info(f"📦 Detected **{len(composite_events)} events**: {clean_events} clean ✓, {composite_count} composite (stacked meals)")

                        with st.expander(f"📦 Event Classification Detail ({len(composite_events)} events)", expanded=False):
                            for comp in composite_events:
                                if comp.is_clean:
                                    st.markdown(f"**{comp.event_id}** (✅ Clean)")
                                    st.markdown(f"- Time: {comp.start_time.strftime('%H:%M')}")
                                    if comp.time_to_peak_minutes:
                                        st.markdown(f"- Time to peak: {comp.time_to_peak_minutes:.0f} min")
                                    if comp.total_glucose_rise:
                                        st.markdown(f"- Glucose rise: **{comp.total_glucose_rise:.0f} mg/dL** ({comp.glucose_at_start:.0f} → {comp.glucose_at_peak:.0f})")
                                else:
                                    st.markdown(f"**{comp.event_id}** (⚡ Composite)")
                                    st.markdown(f"- Start: {comp.start_time.strftime('%H:%M')} | End: {comp.end_time.strftime('%H:%M')}")
                                    st.markdown(f"- Primary event + {len(comp.secondary_events)} secondary event(s)")
                                    if comp.time_to_peak_minutes:
                                        st.markdown(f"- Time to peak: {comp.time_to_peak_minutes:.0f} min")
                                    if comp.total_glucose_rise:
                                        st.markdown(f"- Total glucose rise: **{comp.total_glucose_rise:.0f} mg/dL** ({comp.glucose_at_start:.0f} → {comp.glucose_at_peak:.0f})")
                                    if comp.delta_g_to_next_event is not None:
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
