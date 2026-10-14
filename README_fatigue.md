# When does a starter's fastball start to fade?

A hierarchical Bayesian changepoint model of in-game fastball velocity for MLB starting pitchers (2025–2026 Statcast), and of how temperature, rest, and day/night shift the point where velocity starts to drop.

Notebook: `fatigue_bayesian_changepoint.ipynb`
Predecessor: `fatigue_vs_temperature.ipynb` (a fixed threshold rule plus a stratified Cox model)

---

## Question

Starters lose fastball velocity as a game goes on. Two questions:

1. **Where** does the decline begin, for each pitcher and each start, and how sure are we?
2. **What moves it**: does heat bring fatigue on earlier? Does extra rest delay it?

The Cox notebook answered (2) with a hand-made fatigue rule ("8-pitch average ≥ 1 mph below baseline"). This project **estimates** the fatigue point instead, with full uncertainty, and lets a start that ended before any decline count as information ("τ is later than pitch N") instead of being dropped.

## Data

| Source | What | How |
|---|---|---|
| Statcast (Baseball Savant) via `pybaseball.statcast()` | every pitch, 2025–2026 regular season | cached to `data/statcast_full_{year}.parquet` (reuses `raw_{year}.parquet` if present) |
| MLB Stats API `/schedule?hydrate=weather` | first-pitch temperature, condition (dome / roof closed), day/night | cached to `fatigue/game_weather.parquet` |

**Filtering**
- Starters = the first pitcher for each team (the first two pitchers by `at_bat_number` in each game).
- Starts with ≥ 50 pitches.
- Each start's **primary fastball** only (whichever of FF / SI he threw most that day), from pitch 10 on (warm-up excluded), with ≥ 15 fastballs.
- Outdoor games only (`INCLUDE_INDOOR = False`), since a dome holds the temperature fixed.
- Development subset: the 40 pitchers with the most starts (`N_PITCHERS = 40`).

Development run: **2,008 starts, 40 pitchers, 62,621 fastballs.**

## Model

For pitch *i* in start *s* by pitcher *j*:

**Initial model**

```math
v_i = \text{base}_s - \text{slope}_j \cdot h(\text{pc}_i - \tau_s) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma)
```

```math
\tau_s = \tau_j + X_s \beta + \sigma_{\tau,\text{start}} \, z_s
```

**Add drift**

```math
v_i = \text{base}_s + \text{drift}_j \, (\text{pc}_i - 40) - \text{slope}_j \cdot h(\text{pc}_i - \tau_s) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma)
```

```math
\tau_s = \tau_j + X_s \beta + \sigma_{\tau,\text{start}} \, z_s
```

- `base_s`: that start's velocity around pitch 40 (`PC_CENTER = 40`), varying around the pitcher's usual velocity.
- `drift_j`: a gentle linear change across the **whole** start (mph per pitch). Centered at pitch 40 so it doesn't distort `base_s`.
- `τ_s`: the **changepoint**, the pitch count where an *extra* decline begins.
- `slope_j`: the extra decline rate after τ (log-normal, so always positive).
- `h(x) = 3·softplus(x/3)`: a smooth hinge, ≈ 0 before τ and ≈ linear after. A hard `max(0, x)` has no gradient at the corner, and NUTS struggles with it.

**Hierarchy (partial pooling).** `τ_j`, `slope_j` and `drift_j` are drawn from league distributions whose means and spreads are estimated from all pitchers, so a pitcher with few starts is pulled toward the league and one with many starts stands on his own data. `base_j` has a fixed, wide prior instead, since each pitcher's velocity level is well determined from his own starts. Everything uses the non-centered form (`mean + sd × z`).

## Detailed version

**Start level** (one value per start *s*):

```math
\text{base}_s = \text{base}_j + \sigma_{\text{base}} \, z^{\text{base}}_s
```

```math
\tau_s = \tau_j + X_s \beta + \sigma_{\tau,\text{start}} \, z^{\tau}_s
```

**Pitcher level** (one value per pitcher *j*):

```math
\text{base}_j \sim \mathcal{N}(\bar{v}, 5)
```

```math
\tau_j = \mu_\tau + \sigma_{\tau,\text{pitcher}} \, z^{\tau}_j
```

```math
\text{slope}_j = \exp\left(\mu_{\log \text{slope}} + \sigma_{\log \text{slope}} \, z^{\text{slope}}_j\right)
```

```math
\text{drift}_j = \mu_{\text{drift}} + \sigma_{\text{drift}} \, z^{\text{drift}}_j
```

All $z \sim \mathcal{N}(0, 1)$ (non-centered form).

All $z \sim \mathcal N(0, 1)$ (non-centered form).

- `base_s`: that start's velocity around pitch 40 (`PC_CENTER = 40`), varying around the pitcher's usual velocity `base_j` by about σ_base ≈ 0.7 mph.
- `base_j`: the pitcher's usual velocity; prior centered on the league-average fastball velocity $\bar v$.
- `drift_j`: a gentle linear change across the **whole** start (mph per pitch). Centered at pitch 40 so it doesn't distort `base_s`.
- `τ_j`: the pitcher's typical changepoint. `τ_s`: that start's changepoint, the pitch count where an *extra* decline begins, shifted by temperature, rest and day/night (`X_s β`).
- `slope_j`: the extra decline rate after τ (log-normal, so always positive).
- `h(x) = 3·softplus(x/3)`: a smooth hinge, ≈ 0 before τ and ≈ linear after. A hard `max(0, x)` has no gradient at the corner, and NUTS struggles with it.

**Hierarchy (partial pooling).** `τ_j`, `slope_j` and `drift_j` are drawn from league distributions whose means and spreads are estimated from all pitchers, so a pitcher with few starts is pulled toward the league and one with many starts stands on his own data. `base_j` has a fixed, wide prior, Normal($\bar v$, 5), instead: each pitcher has dozens of starts, so his velocity level is well determined from his own data.

**Three levels of the changepoint**
```text
League:    μ_τ ≈ 83                              (all pitchers)
              │  + pitcher-to-pitcher variation
              ▼
Pitcher:   τ_j = μ_τ + σ_τ,pitcher · z_j          e.g. Pitcher A ≈ 95, Pitcher B ≈ 70
              │  + game conditions + start-to-start variation
              ▼
Start:     τ_s = τ_j + X_s β + σ_τ,start · z_s    e.g. A's start on a hot day ≈ 101
```

**Start-level covariates** (centered, so `τ_j` = a 70°F night game on normal rest):

| Covariate | Coding |
|---|---|
| `temp_10F` | (°F − 70) / 10 |
| `extra_rest` | rest days − 5, clipped to −1…3 |
| `day_game` | 1 = day game |

**Priors**

| Parameter | Prior | Meaning |
|---|---|---|
| `mu_tau` | Normal(85, 20) | league changepoint around 85 pitches |
| `sd_tau_pitcher`, `sd_tau_start` | HalfNormal(15) | pitcher-to-pitcher and start-to-start spread of τ |
| `beta` | Normal(0, 10) | covariate effects, in pitches |
| `mu_logslope`, `sd_logslope` | Normal(log 0.03, 0.7), HalfNormal(0.5) | ≈ 1 mph per 30 pitches after τ |
| `mu_drift`, `sd_drift` | Normal(0, 0.01), HalfNormal(0.01) | ± 0.3 mph per 30 pitches; narrow on purpose |
| `sd_base_start` | HalfNormal(1) | start-to-start velocity spread |
| `sigma` | HalfNormal(1.5) | pitch-to-pitch noise |

**Why drift was added.** Version 1 had no drift term and no way to draw a slow all-game slide except by putting the hinge *before pitch 0*, which gave several pitchers negative changepoints. With `drift_j` in the model, a steady slide is drift and τ means a real change in the rate. `month` was removed at the same time (P(β<0) = 0.24, inconclusive, and confounded with temperature).



**Inference.** NumPyro NUTS, 2 chains × (1,000 warmup + 750 draws), `target_accept_prob = 0.9`, `init_to_median`. `XLA_FLAGS=--xla_force_host_platform_device_count=2` runs the chains in parallel on CPU.

## Code: the drift model

```python
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

SMOOTH = 3.0       # hinge width, pitches
PC_CENTER = 40.0   # drift centered here, so base_start = velocity near pitch 40
COVS = ["temp_10F", "extra_rest", "day_game"]

def hinge(x):
    return SMOOTH * jax.nn.softplus(x / SMOOTH)

def fatigue_model(pc, start_idx, pitcher_of_start, X, n_starts, n_pitchers, v_mean, v=None):
    # League level
    mu_tau = numpyro.sample("mu_tau", dist.Normal(85.0, 20.0))
    sd_tau_pitcher = numpyro.sample("sd_tau_pitcher", dist.HalfNormal(15.0))
    sd_tau_start = numpyro.sample("sd_tau_start", dist.HalfNormal(15.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 10.0).expand([X.shape[1]]).to_event(1))
    mu_logslope = numpyro.sample("mu_logslope", dist.Normal(jnp.log(0.03), 0.7))
    sd_logslope = numpyro.sample("sd_logslope", dist.HalfNormal(0.5))
    mu_drift = numpyro.sample("mu_drift", dist.Normal(0.0, 0.01))
    sd_drift = numpyro.sample("sd_drift", dist.HalfNormal(0.01))
    sd_base_start = numpyro.sample("sd_base_start", dist.HalfNormal(1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.5))

    # Pitcher level
    with numpyro.plate("pitchers", n_pitchers):
        tau_p = numpyro.deterministic(
            "tau_pitcher", mu_tau + sd_tau_pitcher * numpyro.sample("z_tau_p", dist.Normal(0, 1)))
        slope_p = numpyro.deterministic(
            "slope_pitcher", jnp.exp(mu_logslope + sd_logslope * numpyro.sample("z_slope_p", dist.Normal(0, 1))))
        drift_p = numpyro.deterministic(
            "drift_pitcher", mu_drift + sd_drift * numpyro.sample("z_drift_p", dist.Normal(0, 1)))
        base_p = numpyro.sample("base_pitcher", dist.Normal(v_mean, 5.0))

    # Start level
    with numpyro.plate("starts", n_starts):
        tau_s = numpyro.deterministic(
            "tau_start", tau_p[pitcher_of_start] + X @ beta
            + sd_tau_start * numpyro.sample("z_tau_s", dist.Normal(0, 1)))
        base_s = numpyro.deterministic(
            "base_start", base_p[pitcher_of_start]
            + sd_base_start * numpyro.sample("z_base_s", dist.Normal(0, 1)))

    # Pitch level
    j = pitcher_of_start[start_idx]
    mu = (base_s[start_idx]
          + drift_p[j] * (pc - PC_CENTER)                  # gradual change all game
          - slope_p[j] * hinge(pc - tau_s[start_idx]))     # extra drop after the changepoint
    with numpyro.plate("pitches", pc.shape[0]):
        numpyro.sample("v", dist.Normal(mu, sigma), obs=v)

kernel = NUTS(fatigue_model, init_strategy=init_to_median, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=750, num_chains=2)
mcmc.run(jax.random.PRNGKey(42), **DATA, v=V)
```

## Results (development run: 40 pitchers, 2,008 starts)

**League curve**
- Changepoint: **83 pitches** (90% interval 72–94) for a 70°F night game on normal rest.
- Drift before it: **−0.02 mph per 30 pitches** (−0.09 to +0.05), essentially flat for the league. Pitchers do differ (`sd_drift` ≈ 0.3 mph per 30 pitches), which is why the term is worth keeping.
- Extra decline after it: **≈ 0.70 mph per 30 pitches**.

**Covariate effects on the changepoint**

| Covariate | Mean (pitches) | 90% interval | P(β < 0) |
|---|---|---|---|
| +10°F | **+5.6** | +3.4 to +8.0 | 0.00 |
| +1 rest day | **−5.7** | −8.3 to −3.0 | 1.00 |
| Day game | **+6.9** | +2.4 to +11.7 | 0.00 |

Read literally: hotter games and day games push the decline **later**, and extra rest pulls it **earlier**. All three run against baseball intuition, so treat them as associations to explain, not causal effects (see Limitations).

**Diagnostics**
- r̂ ≤ 1.01 for all hyperparameters; n_eff ≥ 120 (lowest: `mu_tau`, 121). Fine for a development run; use more draws for the final run.
- Posterior predictive: residual SD **0.89 mph** vs. posterior σ **0.91 mph**; **90.8%** of real pitches fall in the 90% predictive interval (target 90%).

Residual SD (0.90 mph) is measured from your data.
Posterior sigma (0.92 mph) is the model’s own estimate of that noise.
If the residual SD were clearly larger than σ, the model would be missing structure and pushing it into “noise”.
If it were clearly smaller, the model would be overfitting, chasing individual pitches.
But they are almost blind to shape problems.

- Mean residual by pitch count shows a **U-shape**: the model under-predicts late (90+ pitches), so the straight-line decline after τ is too steep. Real velocity loss flattens out.
- Spread of τ is large: `sd_tau_pitcher` ≈ 35 and `sd_tau_start` ≈ 45 pitches. Per-start changepoints are weakly identified, especially in starts that end near τ, so the per-start τ posteriors are wide. Pitcher-level and league-level quantities are much better determined.

## Limitations and confounds

- **In-game cooling.** First-pitch temperature is fixed for the whole game, but night games cool as they go. On a cool night the late innings are colder still, which can lower late velocity and make τ look earlier. That could produce both the "heat delays fatigue" and the "day games delay fatigue" results. Hourly temperature would separate them.
- **Rest days.** `rest_days` is a plain difference between a pitcher's consecutive starts, so season openers, IL returns, and spells in the bullpen or minors look like "extra rest". Those starts are often shorter and managed differently, which could explain the negative rest effect. Fix: compute within pitcher-season and add a `long_layoff` flag.
- **Survivor bias.** Managers pull pitchers who are fading, so starts that reach 100+ pitches are the ones that held up. The late part of the curve is estimated from a selected group.
- **Decline shape.** A linear hinge after τ is too simple (the residual U-shape above).
- **Subset.** 40 workhorse starters, not a random sample of all starters.
- **Fastball only.** A pitcher who shifts his mix late isn't captured; spin and release height may fade on a different schedule.

## Next steps

1. **Saturating decline + first-inning bump:** replace `slope × h(pc − τ)` with `drop × (1 − exp(−h(pc − τ)/λ))` and add `bump_j × exp(−(pc − 10)/8)`. Check that the residual U-shape is gone.
2. Fix `rest_days` (within pitcher-season, `long_layoff` flag).
3. Hourly park temperature (Open-Meteo) and dew point, plus same-park roof-open vs. roof-closed comparisons.
4. Scale to all starters (`N_PITCHERS = None`), using SVI (`AutoNormal`) first if NUTS is too slow.
5. Joint model of velocity, spin, and release height sharing one changepoint.

## How to run

```bash
python -m venv bayes-env
source bayes-env/bin/activate            # Windows: bayes-env\Scripts\activate
python -m pip install numpyro jax pybaseball pyarrow pandas matplotlib requests ipykernel
python -m ipykernel install --user --name bayes-env
```

## Notebook map

1. Setup and load Statcast; build starts and primary fastballs
2. Game-time weather
3. Covariates and model inputs
4. Model and prior predictive check
5. NUTS fit and convergence summary
6. Results: covariate effects, pitcher profiles (drift + changepoint forest plot), early-changepoint diagnostic, individual starts, P(fatigued) by pitch count at 55 / 75 / 95°F, posterior predictive and residual checks
7. Comparison with the old threshold rule
