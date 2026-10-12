# Baseball Analytics Projects

MLB analytics projects built on Statcast (via `pybaseball`), the MLB Stats API, Baseball Savant leaderboards, and Driveline's OpenBiomechanics data. Each notebook runs on its own in VS Code / Jupyter.

## Projects

| Project | What it does | Main tools |
|---|---|---|
| [Fatigue changepoint](fatigue_bayesian_changepoint.ipynb) | Hierarchical Bayesian changepoint model of in-game fastball velocity: estimates the pitch count where each starter's velocity starts to fade, and how temperature, rest and day/night shift it. | NumPyro, JAX |
| [Fatigue vs. temperature](fatigue_vs_temperature.ipynb) | Earlier version of the fatigue question: a fixed fatigue rule (8-pitch average ≥ 1 mph below baseline) analyzed with Kaplan–Meier curves and a pitcher-stratified Cox model. | statsmodels |
| [Catcher framing](catcher_framing_bayesian.ipynb) | Catcher framing runs: a gradient-boosting model sets each pitch's strike probability, then a hierarchical logistic model separates catcher, pitcher, batter and umpire effects. Compares 2025 vs. 2026 (ABS challenge system) and Baseball Savant. | scikit-learn, NumPyro |
| [Pitch classification](arsenal_classification_lightgbm_transformer.ipynb) | Pitch-type classification from pitch physics, using features relative to each pitcher's own fastball; LightGBM baseline and a transformer over a pitcher's arsenal. | LightGBM, PyTorch |
| [Pitch visualization](pitch_data_visualization.ipynb) | Exploratory plots: feature distributions by pitch type, look-alike cutters vs. sliders, and four-seam / sinker / cutter movement. | pandas, matplotlib, seaborn |
| [Pitching biomechanics](obp_full_signal_and_c3d.ipynb) | OpenBiomechanics data (411 pitches): kinematic sequence, joint angles and energy flow for fast vs. slow throwers, plus 3D stick-figure animations from raw C3D markers. | ezc3d, matplotlib |
| [True whiff rate](bayesian_model_True_Whiff_Rate.ipynb) | Bayesian estimate of pitchers' true whiff rate, shrinking small samples toward the league average. | |
| [Pitch count time series](pitch_count_time_series.ipynb) | Time-series analysis of pitch-level data across a game. | |

Detailed write-ups: [`README_fatigue.md`](README_fatigue.md) · [`README_framing.md`](README_framing.md)

## Key results

- **Fatigue:** league changepoint at about **83 pitches** (90% CI 72–94), followed by about **0.7 mph** lost per 30 pitches; 90.8% of pitches fell inside the 90% predictive interval.
- **Framing:** top 2025 framer Patrick Bailey was worth **+19.6 runs** (90% CI 15.3–23.9); catcher spread did not shrink in 2026 (σ 0.148 → 0.165).

## Data sources

- **Statcast** pitch-by-pitch data, 2025–2026, via `pybaseball.statcast()`
- **MLB Stats API:** game weather, day/night, home-plate umpires
- **Baseball Savant:** catcher framing leaderboard (`catcher-framing_savant.csv`)
- **OpenBiomechanics** (Driveline): full-signal tables and C3D motion capture

## Setup

```bash
python -m venv bayes-env
source bayes-env/bin/activate
pip install pybaseball pandas numpy matplotlib seaborn scikit-learn lightgbm torch numpyro jax arviz statsmodels ezc3d pyarrow requests ipykernel
```

Statcast downloads are cached as parquet files on first run, so later runs are fast.

## Data Visualization

**Process and Analysis(Simple):**        
https://docs.google.com/spreadsheets/d/1l9Aji8jOUwCO1sdpdPhoBPiFYn_QzAKk5FTwtRjfboo/edit?usp=sharing

**Pitching biomechanics:**       
<img src="Pitch_Stick_Flow.gif" width="600">


