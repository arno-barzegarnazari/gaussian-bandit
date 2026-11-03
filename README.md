# Gaussian Thompson Sampling

Roman Ghukasyan | DS223 Marketing Analytics

A Streamlit application comparing alternatives with Bayesian bandits. The original carousel pages use click/no-click feedback. The Gaussian Bandit page extends the model to continuous rewards with known noise standard deviation.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. From the repository root:

```bash
uv sync --locked
uv run streamlit run main.py
```

The app contains four pages: Create Carousels, Interaction, Analytics, and Gaussian Bandit. It runs entirely in the frontend without a database or API keys. Dependencies are pinned in `uv.lock`.

## Original carousel model

Each alternative starts with a Beta(1, 1) prior for its unknown click probability. Each round, the app draws one possible click probability from every current Beta posterior. It ranks these samples and displays the highest ones, up to the selected visible area. This is Thompson sampling: alternatives can be selected because they look promising or because uncertainty gives them a high sample.

Submitting feedback gives every displayed alternative one impression. For the clicked alternative, alpha and clicks increase by one. For each displayed alternative that was not clicked, beta increases by one. With no click, beta increases for every displayed alternative. Alternatives outside the display do not change. For a displayed alternative with binary reward r:

$$\alpha' = \alpha + r, \qquad \beta' = \beta + 1-r.$$

The posterior estimated click probability is `alpha / (alpha + beta)`. History records the displayed alternatives, clicked alternative, timestamp, and round reward: 1 for a click and 0 for no click. The app then samples a new display. Resampling without feedback does not update the posteriors.

`st.session_state.carousels` holds the alternative tables, displays, and histories. `active_carousel_id` connects the selected carousel across pages. Create Carousels creates the state, Interaction updates it, and Analytics reads it. The state is limited to the current browser session. Analytics reports clicks divided by impressions, so its average reward is a per-impression click rate.

## Gaussian model

There are three alternatives. Selecting alternative i produces one continuous reward:

$$R \mid A=i \sim \mathcal{N}(\mu_i,\sigma^2).$$

The reward standard deviation sigma is known and shared across alternatives. Each mean is unknown to the learner and has the same independent prior:

$$\mu_i \sim \mathcal{N}(0,25).$$

The prior variance is 25, so its standard deviation is 5. Each round samples a possible mean from each current Normal posterior, selects the largest sample, generates one reward, and updates only the selected alternative. Unselected alternatives keep their previous posteriors.

`select_alternative()` receives only learned posterior means, posterior variances, and a random generator. True means are used to simulate rewards and evaluate performance. Negative and fractional rewards are preserved.

### Posterior update

For the selected alternative, let m and v be its current posterior mean and variance, and r its observed reward:

$$v_{new}=\left(\frac{1}{v}+\frac{1}{\sigma^2}\right)^{-1},$$

$$m_{new}=v_{new}\left(\frac{m}{v}+\frac{r}{\sigma^2}\right).$$

The update combines the old estimate and new reward using their precisions. Reward precision is `1 / sigma**2`. Known sigma means reward noise is given, rather than estimated. Posterior standard deviation measures uncertainty about the mean and decreases as that alternative receives observations.

Selection samples from Normal(m, v), using `sqrt(v)` as the standard deviation. The final recommendation uses the largest posterior mean, rather than a random sample or the selection count. Exact ties use the lowest alternative ID.

### Interface

The Gaussian Bandit page accepts three true means, sigma, rounds, and seed. It displays each true mean, posterior estimate, posterior standard deviation, and selection count, plus total observed reward and the final recommendation. Charts show learning, selections, and rewards; summary and history CSVs can be downloaded.

Every Run starts from the original prior and a fresh NumPy PCG64 generator. Results remain in a separate session-state entry during sidebar navigation and do not affect the carousel pages. Sigma must be finite and positive, and rounds must be a positive integer. Practical limits are 100,000 rounds, sigma from 0.000001 to 1,000,000, means within +/-1,000,000, and integer seeds from 0 to 4,294,967,295.

## Experiments

All experiments use true means **[1, 2, 3]**, the same **Normal(0, 25)** priors, and **seed 42**. Values below are rounded; [summary.csv](experiments/summary.csv) contains the full-precision results.

| Experiment | Sigma | Rounds | Alternative | Posterior mean | Posterior SD | Selections |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Short run | 1 | 100 | 1 | 1.5702 | 0.4975 | 4 |
| Short run | 1 | 100 | 2 | 1.7287 | 0.7001 | 2 |
| Short run | 1 | 100 | 3 | 2.9950 | 0.1031 | 94 |
| Longer run | 1 | 1,000 | 1 | 1.3062 | 0.4454 | 5 |
| Longer run | 1 | 1,000 | 2 | 2.1421 | 0.2769 | 13 |
| Longer run | 1 | 1,000 | 3 | 2.9792 | 0.0319 | 982 |
| Higher noise | 3 | 1,000 | 1 | 1.1537 | 0.6491 | 21 |
| Higher noise | 3 | 1,000 | 2 | 1.4838 | 0.6207 | 23 |
| Higher noise | 3 | 1,000 | 3 | 2.9542 | 0.0970 | 956 |

| Experiment | Total observed reward | Recommendation | Alternative 3 selection share |
| --- | ---: | --- | ---: |
| Short run | 291.5184 | Alternative 3 | 94.0% |
| Longer run | 2,960.2015 | Alternative 3 | 98.2% |
| Higher noise | 2,884.6044 | Alternative 3 | 95.6% |

### Discussion

With 100 rounds, Alternative 3 is already selected 94 times. Alternatives 1 and 2 have only four and two observations, so their estimates remain uncertain.

With 1,000 rounds at the same noise level, Alternative 3 receives 982 selections. Its posterior SD decreases from 0.1031 to 0.0319. More observations provide stronger evidence about its mean, but the alternatives do not receive equal amounts of data.

An unexpected result is that Alternative 3's estimate is closer to 3 after 100 rounds than after 1,000 rounds: 2.9950 versus 2.9792. Random rewards can move an estimate away from the truth even as posterior uncertainty falls. More observations do not guarantee a smaller realized estimation error at every step.

Increasing sigma from 1 to 3 increases reward variance from 1 to 9. Each reward then contributes one ninth of the observation precision. In this run, the algorithm explores Alternatives 1 and 2 more often, Alternative 3's selection share falls to 95.6%, and its posterior SD increases to 0.0970 compared with the low-noise longer run. The recommendation remains Alternative 3.

Alternative 2's higher-noise estimate, 1.4838, is below its true mean of 2 despite having 23 observations. These rewards are noisier, their realized average is low, and the prior also pulls the estimate toward zero. A larger count under higher noise does not guarantee a more accurate realized estimate. The higher-noise run contains 191 negative rewards, all kept unchanged.

All three recommendations are correct for this seed. These single-seed experiments illustrate learning behavior; they do not guarantee the same outcome for every seed.

### Reproducibility

The experiment script regenerates the summary and checks exact repeatability of complete histories, selection-count totals, and agreement with an independent closed-form posterior calculation:

```bash
uv run python experiments/run_experiments.py
```

Repeating each identical configuration produces the same history. Selection counts sum to 100, 1,000, and 1,000 respectively. For n observations of one alternative, the independently checked posterior is:

$$v_n=\left(\frac{1}{25}+\frac{n}{\sigma^2}\right)^{-1},\qquad m_n=v_n\frac{\sum_{j=1}^{n}r_j}{\sigma^2}.$$

The round history also includes cumulative pseudo-regret, the sum of the best true mean minus the selected true mean. This is an evaluation statistic and is never used for selection.

## Deployment

Streamlit Community Cloud configuration:

| Setting | Value |
| --- | --- |
| Branch | `main` |
| Entrypoint | `main.py` |
| Python | `3.12` |
| Dependencies | Root `uv.lock` and `pyproject.toml` |

## Credits

The original frontend was adapted from [hovhannisyan91/DS223_Group_Project](https://github.com/hovhannisyan91/DS223_Group_Project/tree/main/app/front), commit `3573a04fc6329a3f08a268b72093097d917c4156`. Its original three pages and Beta-Bernoulli algorithm are retained. Gaussian sampling is implemented in [gaussian_bandit.py](gaussian_bandit.py) and [the new page](pages/Gaussian_Bandit.py). The existing carousel Delete button also removes the selected carousel from session state.

- [Application entrypoint](main.py)
- [Experiment results](experiments/summary.csv)
- [Original carousel application](https://carousel-mvp.streamlit.app/)
