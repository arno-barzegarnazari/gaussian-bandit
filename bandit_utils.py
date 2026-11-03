# bandit_utils.py

import random
import uuid
from datetime import datetime
from math import lgamma
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta as scipy_beta


DEFAULT_ALPHA = 1
DEFAULT_BETA = 1


def init_state() -> None:
    if "carousels" not in st.session_state:
        st.session_state.carousels = {}
        # TODO: Load carousels from API: GET /carousels

    if "active_carousel_id" not in st.session_state:
        st.session_state.active_carousel_id = None


def create_alternatives_df(total_alternatives: int) -> pd.DataFrame:
    rows = []
    for i in range(1, total_alternatives + 1):
        rows.append(
            {
                "alt_id": i,
                "title": f"Alternative {i}",
                "alpha": DEFAULT_ALPHA,
                "beta": DEFAULT_BETA,
                "impressions": 0,
                "clicks": 0,
            }
        )
    return pd.DataFrame(rows)


def create_carousel(name: str, total_alternatives: int, visible_count: int) -> str:
    carousel_id = str(uuid.uuid4())[:8]
    st.session_state.carousels[carousel_id] = {
        "carousel_id": carousel_id,
        "name": name,
        "total_alternatives": total_alternatives,
        "visible_count": visible_count,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "round_number": 1,
        "alternatives": create_alternatives_df(total_alternatives),
        "history": [],
        "current_display_ids": [],
    }
    st.session_state.active_carousel_id = carousel_id
    # TODO: Persist carousel to database via API: POST /carousels with carousel data
    return carousel_id


def get_carousel(carousel_id: str) -> Optional[dict]:
    return st.session_state.carousels.get(carousel_id)


def get_carousel_options() -> Dict[str, str]:
    return {
        cid: f"{data['name']} ({data['visible_count']}/{data['total_alternatives']})"
        for cid, data in st.session_state.carousels.items()
    }


def get_stats_df(carousel_id: str) -> pd.DataFrame:
    carousel = get_carousel(carousel_id)
    df = carousel["alternatives"].copy()
    df["posterior_mean"] = df["alpha"] / (df["alpha"] + df["beta"])
    df["empirical_ctr"] = df.apply(
        lambda row: row["clicks"] / row["impressions"] if row["impressions"] > 0 else 0.0,
        axis=1,
    )
    return df


def get_champion_alt_id(carousel_id: str) -> Optional[int]:
    stats = get_stats_df(carousel_id).sort_values(
        ["posterior_mean", "clicks", "impressions", "alt_id"],
        ascending=[False, False, False, True],
    )
    if stats.empty:
        return None
    return int(stats.iloc[0]["alt_id"])


def beta_pdf_points(alpha: int, beta: int, num_points: int = 200) -> pd.DataFrame:
    x = np.linspace(0.001, 0.999, num_points)
    if scipy_beta is not None:
        y = scipy_beta.pdf(x, alpha, beta)
    else:
        log_norm = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
        y = np.exp(log_norm + (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x))
    return pd.DataFrame({"theta": x, "density": y})


def sample_carousel(carousel_id: str) -> None:
    carousel = get_carousel(carousel_id)
    stats = get_stats_df(carousel_id)

    sampled = []
    for _, row in stats.iterrows():
        theta = random.betavariate(int(row["alpha"]), int(row["beta"]))
        sampled.append((int(row["alt_id"]), theta))

    sampled.sort(key=lambda x: x[1], reverse=True)
    chosen = [alt_id for alt_id, _ in sampled[: carousel["visible_count"]]]
    carousel["current_display_ids"] = chosen


def get_current_display_df(carousel_id: str) -> pd.DataFrame:
    carousel = get_carousel(carousel_id)
    if not carousel["current_display_ids"]:
        sample_carousel(carousel_id)

    ids = carousel["current_display_ids"]
    stats = get_stats_df(carousel_id)
    view = stats[stats["alt_id"].isin(ids)].copy()
    view["slot"] = view["alt_id"].apply(lambda x: ids.index(x) + 1)
    view = view.sort_values("slot").reset_index(drop=True)
    return view


def update_after_feedback(carousel_id: str, clicked_alt_id=None) -> None:
    carousel = get_carousel(carousel_id)
    displayed_ids = carousel["current_display_ids"]

    if not displayed_ids:
        sample_carousel(carousel_id)
        displayed_ids = carousel["current_display_ids"]

    df = carousel["alternatives"].copy()

    for alt_id in displayed_ids:
        idx = df.index[df["alt_id"] == alt_id][0]
        df.at[idx, "impressions"] += 1

        if clicked_alt_id is not None and alt_id == clicked_alt_id:
            df.at[idx, "alpha"] += 1
            df.at[idx, "clicks"] += 1
        else:
            df.at[idx, "beta"] += 1

    carousel["alternatives"] = df
    carousel["history"].append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "round": carousel["round_number"],
            "displayed": displayed_ids.copy(),
            "clicked": clicked_alt_id,
            "reward": 1 if clicked_alt_id is not None else 0,
        }
    )
    carousel["round_number"] += 1
    sample_carousel(carousel_id)
    # TODO: Update carousel and alternatives in database via API: PUT /carousels/{carousel_id} and POST /interactions


def reset_carousel(carousel_id: str) -> None:
    carousel = get_carousel(carousel_id)
    carousel["alternatives"] = create_alternatives_df(carousel["total_alternatives"])
    carousel["history"] = []
    carousel["current_display_ids"] = []
    carousel["round_number"] = 1
    sample_carousel(carousel_id)
    # TODO: Reset carousel in database via API: PUT /carousels/{carousel_id}/reset


def carousels_table() -> pd.DataFrame:
    rows = []
    for cid, c in st.session_state.carousels.items():
        stats = get_stats_df(cid)

        total_impressions = int(stats["impressions"].sum())
        total_clicks = int(stats["clicks"].sum())
        avg_reward = total_clicks / total_impressions if total_impressions else 0.0

        rows.append(
            {
                "carousel_id": cid,
                "name": c["name"],
                "visible_count": c["visible_count"],
                "total_alternatives": c["total_alternatives"],
                "created_at": c["created_at"],
                "round_number": c["round_number"],
                "total_impressions": total_impressions,
                "total_clicks": total_clicks,
                "average_reward": f"{avg_reward:.2%}",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "carousel_id",
                "name",
                "visible_count",
                "total_alternatives",
                "created_at",
                "round_number",
                "total_impressions",
                "total_clicks",
                "average_reward",
            ]
        )

    return pd.DataFrame(rows)


def select_carousel_widget(key_suffix: str = "") -> Optional[str]:
    options = get_carousel_options()
    if not options:
        st.info("Create a carousel first on the Create Carousels page.")
        return None

    ids = list(options.keys())
    default_index = 0

    if st.session_state.active_carousel_id in ids:
        default_index = ids.index(st.session_state.active_carousel_id)

    selected_id = st.selectbox(
        "Choose carousel",
        ids,
        index=default_index,
        format_func=lambda x: options[x],
        key=f"carousel_select_{key_suffix}",
    )

    st.session_state.active_carousel_id = selected_id
    return selected_id
