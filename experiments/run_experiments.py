"""Regenerate the three required experiments from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from gaussian_bandit import run_simulation


def main():
    output = Path(__file__).resolve().parent
    summaries = []
    for name, sigma, rounds in [("short_run", 1.0, 100), ("longer_run", 1.0, 1000), ("higher_noise", 3.0, 1000)]:
        result = run_simulation([1, 2, 3], sigma, rounds, seed=42)
        repeat = run_simulation([1, 2, 3], sigma, rounds, seed=42)
        pd.testing.assert_frame_equal(result.history, repeat.history, check_exact=True)
        np.testing.assert_array_equal(result.counts, repeat.counts)
        assert int(result.counts.sum()) == rounds
        for i in range(3):
            rewards = result.history.loc[result.history.selected_alternative == i + 1, "reward"]
            expected_variance = 1.0 / (1.0 / 25.0 + len(rewards) / sigma**2)
            expected_mean = expected_variance * rewards.sum() / sigma**2
            np.testing.assert_allclose(result.posterior_means[i], expected_mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(result.posterior_variances[i], expected_variance, rtol=1e-12)
        summary = result.summary()
        summary.insert(0, "experiment", name)
        summary["sigma"] = sigma
        summary["rounds"] = rounds
        summary["seed"] = 42
        summary["total_observed_reward"] = result.total_reward
        summary["recommendation"] = result.recommendation
        summary["cumulative_pseudo_regret"] = result.history.cumulative_pseudo_regret.iloc[-1]
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output / "summary.csv", index=False)
    print(combined.to_string(index=False))
    print("\nReproducibility, count totals and closed-form posterior checks passed for all three runs.")


if __name__ == "__main__":
    main()
