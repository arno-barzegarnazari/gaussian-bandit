# pages/1_Create_Carousels.py

import streamlit as st

from bandit_utils import (
    carousels_table,
    create_carousel,
    get_carousel,
    init_state,
    sample_carousel,
    reset_carousel,
    select_carousel_widget,
)

init_state()

st.title("Create Carousels")
st.caption("Start here to create a carousel project and choose how many bandits are visible per round.")

with st.form("create_carousel_form", clear_on_submit=True):
    name = st.text_input(
        "Carousel name",
        value=f"Carousel {len(st.session_state.carousels) + 1}",
    )
    total_alternatives = st.number_input(
        "Total alternatives",
        min_value=2,
        max_value=100,
        value=10,
        step=1,
    )
    visible_count = st.number_input(
        "Visible area (how many bandits to show)",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
    )

    submitted = st.form_submit_button("Create carousel")

    if submitted:
        total_alternatives = int(total_alternatives)
        visible_count = int(visible_count)

        if visible_count > total_alternatives:
            st.error("Visible area cannot be greater than total alternatives.")
        else:
            carousel_id = create_carousel(name, total_alternatives, visible_count)
            sample_carousel(carousel_id)
            st.success("Carousel created successfully.")

st.subheader("Available carousels")
df = carousels_table()

if df.empty:
    st.info("No carousels created yet.")
else:
    st.dataframe(df, width="stretch", hide_index=True)

    selected_id = select_carousel_widget("manage")

    if selected_id:
        carousel = get_carousel(selected_id)

        col1, col2, col3 = st.columns(3)
        col1.metric("Visible area", carousel["visible_count"])
        col2.metric("Total alternatives", carousel["total_alternatives"])
        col3.metric("Current round", carousel["round_number"])

        st.markdown("### Alternatives")
        st.dataframe(carousel["alternatives"], width="stretch", hide_index=True)

        if st.button("Reset selected carousel"):
            reset_carousel(selected_id)
            st.success("Selected carousel reset.")
            st.rerun()

        if st.button("Delete selected carousel", type="secondary"):
            del st.session_state.carousels[selected_id]
            st.session_state.active_carousel_id = next(iter(st.session_state.carousels), None)
            st.success("Selected carousel deleted.")
            st.rerun()
