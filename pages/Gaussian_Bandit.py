"""Interactive Gaussian bandit experiment; independent of carousel state."""
import pandas as pd
import streamlit as st

from gaussian_bandit import MAX_ROUNDS, run_simulation

st.title("Gaussian Bandit")
st.caption("Three alternatives, continuous rewards, and Thompson sampling with known reward noise.")
st.info("Each unknown mean starts with a Normal(0, 25) prior: mean 0, variance 25, standard deviation 5.")

previous = st.session_state.get("gaussian_result")
default_means = previous.true_means if previous is not None else [1.0, 2.0, 3.0]

with st.form("gaussian_config"):
    st.subheader("Simulation settings")
    cols = st.columns(3)
    means = [col.number_input(f"True mean {i + 1}", min_value=-1e6, max_value=1e6,
                             value=float(default_means[i]), step=0.1, key=f"gaussian_mu_{i}")
             for i, col in enumerate(cols)]
    c1, c2, c3 = st.columns(3)
    sigma = c1.number_input("Known noise SD (sigma)", value=previous.sigma if previous is not None else 1.0, step=0.1, format="%.6f")
    rounds = c2.number_input("Number of rounds", min_value=1, max_value=MAX_ROUNDS,
                             value=previous.rounds if previous is not None else 1000, step=1)
    seed = c3.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=previous.seed if previous is not None else 42, step=1)
    st.caption("True means generate simulated rewards. The selection rule only sees the learned posteriors. "
               "Sigma must be positive; negative rewards are kept unchanged.")
    submitted = st.form_submit_button("Run", type="primary")

if submitted:
    try:
        with st.spinner("Running Thompson sampling..."):
            result = run_simulation(means, sigma, rounds, seed)
        st.session_state["gaussian_result"] = result
    except ValueError as error:
        # Avoid showing an older run beneath an error as if it were a new result.
        st.session_state.pop("gaussian_result", None)
        st.error(str(error))

result = st.session_state.get("gaussian_result")
if result is not None:
    st.subheader("Results")
    st.caption(f"Completed run: true means {result.true_means.tolist()}, sigma {result.sigma:g}, "
               f"{result.rounds:,} rounds, seed {result.seed}. Press Run to apply new settings.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total observed reward", f"{result.total_reward:,.3f}")
    c2.metric("Final recommendation", f"Alternative {result.recommendation}")
    c3.metric("Completed rounds", f"{result.rounds:,}")
    st.dataframe(result.summary(), hide_index=True, width="stretch", column_config={
        "True mean": st.column_config.NumberColumn(format="%.4f"),
        "Posterior mean": st.column_config.NumberColumn(format="%.4f"),
        "Posterior SD": st.column_config.NumberColumn(format="%.4f"),
        "Selection share": st.column_config.NumberColumn(format="percent"),
    })
    st.success(f"Selection counts: {int(result.counts.sum()):,} / {result.rounds:,} rounds. "
               "Recommendation uses the largest posterior mean.")
    history = result.history
    tabs = st.tabs(["Learning", "Rewards", "Round history", "Model"])
    with tabs[0]:
        initial = {"round": 0, **{f"posterior_mean_{i}": 0.0 for i in range(1, 4)}}
        learning = pd.concat([pd.DataFrame([initial]), history], ignore_index=True)
        curve = learning[["round"] + [f"posterior_mean_{i}" for i in range(1, 4)]]
        curve = curve.rename(columns={f"posterior_mean_{i}": f"Alternative {i}" for i in range(1, 4)})
        st.markdown("**Posterior means over time**")
        st.line_chart(curve.set_index("round"))
        st.caption("True means for comparison: " + ", ".join(f"A{i + 1} = {m:g}" for i, m in enumerate(result.true_means)))
        st.markdown("**Selection counts**")
        st.bar_chart(result.summary().set_index("Alternative")[["Selections"]])
    with tabs[1]:
        st.markdown("**Cumulative observed reward**")
        st.line_chart(history.set_index("round")[["cumulative_reward"]])
        st.markdown("**Individual continuous rewards**")
        st.line_chart(history.set_index("round")[["reward"]])
        st.caption("Observed reward may be negative. Cumulative reward can decrease.")
        st.metric("Cumulative pseudo-regret (evaluation only)", f"{history['cumulative_pseudo_regret'].iloc[-1]:,.3f}")
        st.caption("Sum of best true mean minus selected true mean. This diagnostic is never used for selection.")
    with tabs[2]:
        st.dataframe(history, hide_index=True, width="stretch")
        st.download_button("Download round history (CSV)", history.to_csv(index=False),
                           "gaussian_history.csv", "text/csv")
        st.download_button("Download summary (CSV)", result.summary().to_csv(index=False),
                           "gaussian_summary.csv", "text/csv")
    with tabs[3]:
        st.markdown("For the selected alternative, let its current posterior be Normal(m, v) and its reward be r.")
        st.latex(r"v_{new} = \left(\frac{1}{v} + \frac{1}{\sigma^2}\right)^{-1}")
        st.latex(r"m_{new} = v_{new}\left(\frac{m}{v} + \frac{r}{\sigma^2}\right)")
        st.write("Each round samples one possible mean from each Normal posterior using standard deviation "
                 "sqrt(v), chooses the largest sample, observes one reward, and updates that alternative only. "
                 "Known sigma describes reward noise; posterior SD describes uncertainty about the mean.")
        st.write("Each Run starts again from the same prior. The seed controls a fresh NumPy PCG64 generator. "
                 "Results stay available during navigation in this browser session.")
else:
    st.write("Choose settings and press Run. For the required experiments use means [1, 2, 3], seed 42, "
             "and (sigma, rounds) = (1, 100), (1, 1000), (3, 1000).")
