

import pandas as pd
import streamlit as st


# st.write('analtycis.page')
from bandit_utils import (
    beta_pdf_points,
    carousels_table,
    get_champion_alt_id,
    get_carousel,
    get_stats_df,
    init_state,
    select_carousel_widget,
)

init_state()

st.title("Analytics")

st.subheader("All carousel projects")
projects_df = carousels_table()
if projects_df.empty:
    st.info("No carousels created yet.")
else:
    st.dataframe(projects_df, width="stretch", hide_index=True)

selected_id = select_carousel_widget("analytics")
if selected_id:
    carousel = get_carousel(selected_id)
    stats = get_stats_df(selected_id).sort_values("posterior_mean", ascending=False).reset_index(drop=True)
    champion_alt_id = get_champion_alt_id(selected_id)
    champion_title = stats.loc[stats["alt_id"] == champion_alt_id, "title"].iloc[0]
    stats["is_champion"] = stats["alt_id"] == champion_alt_id
    stats["series"] = stats["is_champion"].map(lambda x: "Champion" if x else "Other bandit")

    st.subheader(f"Analytics for {carousel['name']}")

    total_impressions = int(stats["impressions"].sum())
    total_clicks = int(stats["clicks"].sum())
    avg_reward = total_clicks / total_impressions if total_impressions else 0.0
    best_alt = champion_title

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rounds", max(carousel["round_number"] - 1, 0))
    k2.metric("Total Impressions", total_impressions)
    k3.metric("Total Clicks", total_clicks)
    k4.metric("Average Reward", f"{avg_reward:.2%}")

    st.write(f"**Champion bandit by posterior mean:** {best_alt}")

    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("**Posterior mean by alternative**")
        st.vega_lite_chart(
            stats,
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "title", "type": "nominal", "sort": None, "title": "Alternative"},
                    "y": {"field": "posterior_mean", "type": "quantitative", "title": "Posterior mean"},
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "scale": {"domain": ["Champion", "Other bandit"], "range": ["#d97706", "#94a3b8"]},
                        "legend": {"title": None, "orient": "top"},
                    },
                    "tooltip": [
                        {"field": "title", "type": "nominal"},
                        {"field": "posterior_mean", "type": "quantitative", "format": ".2%"},
                        {"field": "alpha", "type": "quantitative"},
                        {"field": "beta", "type": "quantitative"},
                    ],
                },
            },
            width="stretch",
        )

    with chart2:
        st.markdown("**Clicks by alternative**")
        st.vega_lite_chart(
            stats,
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "title", "type": "nominal", "sort": None, "title": "Alternative"},
                    "y": {"field": "clicks", "type": "quantitative", "title": "Clicks"},
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "scale": {"domain": ["Champion", "Other bandit"], "range": ["#d97706", "#94a3b8"]},
                        "legend": {"title": None, "orient": "top"},
                    },
                    "tooltip": [
                        {"field": "title", "type": "nominal"},
                        {"field": "clicks", "type": "quantitative"},
                        {"field": "impressions", "type": "quantitative"},
                        {"field": "empirical_ctr", "type": "quantitative", "format": ".2%"},
                    ],
                },
            },
            width="stretch",
        )

    st.markdown("**Beta posterior distributions**")
    beta_curves = []
    for _, row in stats.iterrows():
        curve = beta_pdf_points(int(row["alpha"]), int(row["beta"]))
        curve["title"] = row["title"]
        curve["series"] = "Champion" if bool(row["is_champion"]) else "Other bandit"
        beta_curves.append(curve)

    beta_df = pd.concat(beta_curves, ignore_index=True)
    st.vega_lite_chart(
        beta_df,
        {
            "mark": {"type": "line", "strokeWidth": 3},
            "encoding": {
                "x": {"field": "theta", "type": "quantitative", "title": "CTR / theta"},
                "y": {"field": "density", "type": "quantitative", "title": "Density"},
                "detail": {"field": "title", "type": "nominal"},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "scale": {"domain": ["Champion", "Other bandit"], "range": ["#d97706", "#94a3b8"]},
                    "legend": {"title": None, "orient": "top"},
                },
                "opacity": {
                    "field": "series",
                    "type": "nominal",
                    "scale": {"domain": ["Champion", "Other bandit"], "range": [1.0, 0.35]},
                    "legend": None,
                },
                "tooltip": [
                    {"field": "title", "type": "nominal"},
                    {"field": "theta", "type": "quantitative", "format": ".3f"},
                    {"field": "density", "type": "quantitative", "format": ".3f"},
                ],
            },
        },
        width="stretch",
    )

    st.markdown("**Detailed results table**")
    table = stats[
        [
            "alt_id",
            "title",
            "series",
            "alpha",
            "beta",
            "impressions",
            "clicks",
            "empirical_ctr",
            "posterior_mean",
        ]
    ].copy()
    table["series"] = table["series"].replace({"Other bandit": ""})
    table["empirical_ctr"] = table["empirical_ctr"].map(lambda x: f"{x:.2%}")
    table["posterior_mean"] = table["posterior_mean"].map(lambda x: f"{x:.2%}")
    st.dataframe(table, width="stretch", hide_index=True)

    st.markdown("**Reward over time**")
    if carousel["history"]:
        hist_df = pd.DataFrame(carousel["history"]).copy()
        hist_df["cumulative_reward"] = hist_df["reward"].cumsum()
        hist_df = hist_df[["round", "reward", "cumulative_reward", "clicked", "timestamp"]]

        st.line_chart(hist_df.set_index("round")["cumulative_reward"])
        st.dataframe(hist_df, width="stretch", hide_index=True)
    else:
        st.info("No interaction history yet.")
