"""Gaussian Thompson sampling with a known, shared reward standard deviation."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 25.0
MAX_ROUNDS = 100_000


@dataclass
class BanditResult:
    true_means: np.ndarray
    sigma: float
    rounds: int
    seed: int
    posterior_means: np.ndarray
    posterior_variances: np.ndarray
    counts: np.ndarray
    history: pd.DataFrame

    @property
    def recommendation(self) -> int:
        """One-based alternative ID; exact ties go to the lowest ID."""
        return int(np.argmax(self.posterior_means)) + 1

    @property
    def total_reward(self) -> float:
        return float(self.history["reward"].sum())

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Alternative": [f"Alternative {i}" for i in range(1, 4)],
            "True mean": self.true_means,
            "Posterior mean": self.posterior_means,
            "Posterior SD": np.sqrt(self.posterior_variances),
            "Selections": self.counts,
            "Selection share": self.counts / self.rounds,
        })


def validate_config(true_means, sigma, rounds, seed) -> np.ndarray:
    means = np.asarray(true_means, dtype=float)
    if means.shape != (3,) or not np.isfinite(means).all():
        raise ValueError("Provide exactly three finite true means.")
    if isinstance(sigma, (bool, np.bool_)) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Known sigma must be a finite positive number.")
    # Prevent floating-point overflow/underflow at unrealistic numerical extremes.
    if sigma < 1e-6 or sigma > 1e6 or np.max(np.abs(means)) > 1e6:
        raise ValueError("Use sigma from 0.000001 to 1000000 and means within +/-1000000.")
    if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, Integral) or not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"Rounds must be a positive integer, at most {MAX_ROUNDS}.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    return means.copy()


def select_alternative(posterior_means, posterior_variances, rng) -> tuple[int, np.ndarray]:
    """The selector has no access to true means or reward-generation logic."""
    samples = rng.normal(posterior_means, np.sqrt(posterior_variances))
    return int(np.argmax(samples)), samples


def update_posterior(mean: float, variance: float, reward: float, sigma: float) -> tuple[float, float]:
    """Normal-Normal conjugate update; sigma is reward SD, not variance."""
    prior_precision = 1.0 / variance
    reward_precision = 1.0 / sigma**2
    new_variance = 1.0 / (prior_precision + reward_precision)
    new_mean = new_variance * (prior_precision * mean + reward_precision * reward)
    return float(new_mean), float(new_variance)


def run_simulation(true_means=(1.0, 2.0, 3.0), sigma=1.0, rounds=1000, seed=42) -> BanditResult:
    means = validate_config(true_means, sigma, rounds, seed)
    # A fresh local generator makes every Run independent of earlier UI activity.
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    posterior_means = np.full(3, PRIOR_MEAN)
    posterior_variances = np.full(3, PRIOR_VARIANCE)
    counts = np.zeros(3, dtype=int)
    records = []
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    for round_number in range(1, int(rounds) + 1):
        chosen, samples = select_alternative(posterior_means, posterior_variances, rng)
        reward = float(rng.normal(means[chosen], sigma))
        posterior_means[chosen], posterior_variances[chosen] = update_posterior(
            posterior_means[chosen], posterior_variances[chosen], reward, sigma
        )
        counts[chosen] += 1
        cumulative_reward += reward
        # Evaluation only: expected reward gap, not observed reward loss.
        cumulative_regret += float(np.max(means) - means[chosen])
        record = {
            "round": round_number, "selected_alternative": chosen + 1,
            "reward": reward, "cumulative_reward": cumulative_reward,
            "cumulative_pseudo_regret": cumulative_regret,
        }
        for i in range(3):
            record[f"sampled_mean_{i + 1}"] = float(samples[i])
            record[f"posterior_mean_{i + 1}"] = float(posterior_means[i])
            record[f"posterior_variance_{i + 1}"] = float(posterior_variances[i])
            record[f"count_{i + 1}"] = int(counts[i])
        records.append(record)
    return BanditResult(means, float(sigma), int(rounds), int(seed), posterior_means,
                        posterior_variances, counts, pd.DataFrame(records))
