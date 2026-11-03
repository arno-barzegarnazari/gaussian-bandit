# pages/2_Interaction.py

import streamlit as st

from bandit_utils import (
    get_carousel,
    get_current_display_df,
    get_stats_df,
    init_state,
    sample_carousel,
    select_carousel_widget,
    update_after_feedback,
)

init_state()

st.title("Interaction")

selected_id = select_carousel_widget("interaction")
if selected_id:
    carousel = get_carousel(selected_id)

    if not carousel["current_display_ids"]:
        sample_carousel(selected_id)

    st.caption(
        f"{carousel['name']} • showing {carousel['visible_count']} out of {carousel['total_alternatives']} alternatives"
    )

    stats = get_stats_df(selected_id)
    total_impressions = int(stats["impressions"].sum())
    total_clicks = int(stats["clicks"].sum())
    global_ctr = total_clicks / total_impressions if total_impressions else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Round", carousel["round_number"])
    m2.metric("Total Clicks", total_clicks)
    m3.metric("Global CTR", f"{global_ctr:.2%}")

    display_df = get_current_display_df(selected_id)
    cols = st.columns(carousel["visible_count"])

    for col, (_, row) in zip(cols, display_df.iterrows()):
        with col:
            with st.container(border=True):
                st.markdown(f"### Alternative {int(row['alt_id'])}")
                st.write(row["title"])
                st.caption(f"Slot {int(row['slot'])}")
                st.write(f"Posterior mean: **{row['posterior_mean']:.2%}**")
                st.write(f"Empirical CTR: **{row['empirical_ctr']:.2%}**")
                st.write(f"α = {int(row['alpha'])}, β = {int(row['beta'])}")

                if st.button(
                    "Click",
                    key=f"click_{selected_id}_{int(row['alt_id'])}",
                    width="stretch",
                ):
                    update_after_feedback(selected_id, clicked_alt_id=int(row["alt_id"]))
                    st.rerun()

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        if st.button("No click / Next round", width="stretch"):
            update_after_feedback(selected_id, clicked_alt_id=None)
            st.rerun()

    with c2:
        if st.button("Resample without update", width="stretch"):
            sample_carousel(selected_id)
            st.rerun()

    st.subheader("Latest event")
    if carousel["history"]:
        last_event = carousel["history"][-1]
        st.write(f"**Round:** {last_event['round']}")
        st.write(f"**Displayed:** {last_event['displayed']}")
        st.write(f"**Clicked:** {last_event['clicked'] if last_event['clicked'] is not None else 'No click'}")
        st.write(f"**Reward:** {last_event['reward']}")
        st.write(f"**Timestamp:** {last_event['timestamp']}")
    else:
        st.info("No interaction yet.")
