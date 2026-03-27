# Quant Pipeline Paper — Research Plan & Roadmap
**Repo:** [github.com/ioan-hedea/sequential-decision-making](https://github.com/ioan-hedea/sequential-decision-making)
**Current state:** Working pipeline, backtest results, diagnostic figures, draft paper (March 2026)

---

## 0. Research Status Snapshot

This section converts the roadmap into a status-aware plan. The goal is to
separate what is already in the codebase from what is only in the paper plan or
still needs to be built.

### 0.1 Already Implemented in Code

- RL is framed as a **control layer**, not a raw alpha generator
- Core architecture in code and report: alpha layer, constrained allocator, RL control, hedge layer, execution/cost layer
- Baselines:
  - factor-only
  - equal-weight
  - volatility targeting
  - drawdown-based deleveraging
  - end-to-end RL baseline (PPO)
- Ablations:
  - factor-only
  - alpha stack without RL
  - alpha plus portfolio RL
  - alpha plus hedge RL
  - full pipeline
- Robustness engine:
  - multiple overlapping rolling windows
  - transaction-cost sensitivity
  - rebalance-band sensitivity
  - hedge-intensity sensitivity
  - macro-lag sensitivity
  - reward-function ablation
- Statistical reliability:
  - block-bootstrap confidence intervals
  - pairwise bootstrap significance tables
- Behavioral analysis:
  - regime-conditioned exposure, hedge ratio, cash, turnover, and returns
- Financial realism:
  - lagged macro features
  - explicit no-lookahead timing contract
  - fixed + volatility-scaled + size-sensitive transaction costs
  - long-only allocator
  - turnover penalty
  - position caps / group caps
- Data and state enrichment:
  - FRED macro series including `VIX`, `HY OAS`, `DXY`
  - uncertainty features including alpha dispersion, regime entropy, and IC instability
- Research artifact export:
  - CSV metrics
  - JSON summaries
  - paper-ready tables
  - summary figures

### 0.2 Partially Implemented or Waiting on Final Evidence

- Central hypotheses are now structurally testable, but the paper still needs
  fresh completed runs to report the new evidence cleanly
- H1: RL improves downside risk vs factor-only
- H2: RL outperforms rule-based risk overlays
- H3: Modular RL outperforms end-to-end RL
- The report now discusses these explicitly, but the main results tables still
  need to be refreshed with the latest research-run outputs
- Additional metrics are mixed:
  - worst 5\% daily return / VaR-style tail metric is already present
  - downside deviation is indirectly present through Sortino
  - hedge cost vs benefit is partly available through hedge PnL logs but not yet
    elevated to a first-class results table

### 0.3 Not Yet Implemented

- H4 / RQ4: option-based hedging improves realism and effectiveness
- Real options sleeve:
  - protective put
  - collar
  - optional put spread
- RL choosing hedge **type** in addition to hedge intensity
- Options features:
  - implied volatility term structure
  - IV percentile
  - richer option-cost inputs
- Fully point-in-time fundamentals from SEC filing dates
- Drawdown duration as a reported headline metric
- Formal Sharpe-ratio comparison tests beyond bootstrap summaries

### 0.4 Immediate Paper Upgrade Priority

These are the highest-value next steps for turning the project from a strong
prototype into a stronger research artifact.

1. Finish one full research run with the new evaluation stack and update the
   report tables/figures from actual artifacts.
2. Surface the rule-based and end-to-end baselines as first-class evidence in
   the main paper, not only in generated CSVs.
3. Add a compact status-aware subsection for RQ4 as future work, rather than
   pretending options integration already exists.
4. Promote hedge cost/benefit and drawdown-duration metrics if the current runs
   support them cleanly.

---

## 1. Research Questions

The paper should be organized around three explicit, testable research questions,
with a fourth extension reserved for the next realism upgrade.

### RQ1 — Architecture: Controller vs. End-to-End

> Is reinforcement learning more effective as a modular control layer over financially grounded alpha models than as an end-to-end trading agent?

This is the most distinctive claim of the paper. It positions the work against the dominant FinRL-style literature that treats RL as a monolithic predictor. Testing it requires building an **end-to-end RL baseline** on the same universe and feature set.

### RQ2 — Value Over Simple Rules

> Does the RL control layer add measurable value beyond well-known rule-based risk management heuristics (volatility targeting, drawdown-based deleveraging)?

This is the "is RL actually doing something nontrivial?" test. Without it, a reviewer can always argue that a simple vol-scaling rule would achieve the same drawdown improvement. Testing it requires implementing **two rule-based baselines**.

### RQ3 — Component Attribution

> Which component of RL control — exposure modulation, dynamic hedging, or their joint operation — is the primary driver of the observed risk transformation?

This is already partially addressed by the existing ablation table, but needs to be sharpened. The current results suggest hedging contributes more than portfolio RL alone — that's an important finding that should be stated as a formal result rather than an observation.

### RQ4 — Option-Based Hedging Realism (Stretch Goal)

> Does replacing the stylized convex hedge sleeve with explicit option-based hedging improve realism and downside protection enough to justify the added complexity?

This should be treated as a second-wave research question rather than a blocker
for the current paper. It is high value, but also meaningfully higher effort
than the current stock-and-overlay architecture.

### Working Hypotheses

- **H1:** RL improves downside risk relative to factor-only.
- **H2:** RL outperforms rule-based risk overlays on at least some risk-adjusted metrics.
- **H3:** Modular RL outperforms end-to-end RL in robustness and interpretability.
- **H4:** Option-based hedging improves realism and potentially protection, but
  it remains untested in the current codebase.

---

## 2. Central Thesis (Revised)

Proposed rewrite for the introduction's "defended statement":

> We study whether reinforcement learning is better used as a sequential portfolio control layer than as an end-to-end predictor, and whether this architecture can improve downside behavior while preserving most of the upside of a finance-first alpha stack. The central defended claim is that a factor-anchored RL controller, when layered onto a strong classical alpha engine under explicit cost and timing assumptions, achieves a materially better drawdown profile than passive equity exposure, outperforms simple rule-based risk management on risk-adjusted metrics, and dominates an end-to-end RL approach on both robustness and interpretability.

---

## 3. New Baselines & Comparisons

### 3.1 End-to-End RL Baseline (Required — tests RQ1)

**What:** A standard RL agent (PPO or DQN) that takes raw features as state and directly outputs portfolio weights or discrete allocation actions.

**Design choices:**
- Same feature set as the modular pipeline (factor scores, vol forecasts, regime beliefs, macro inputs)
- Same universe, same backtest period, same transaction-cost model
- Action space: either continuous weights (PPO) or discretized allocation buckets (DQN)
- Reward: same differential Sharpe reward used by the portfolio RL agent
- Training: same walk-forward protocol

**Why it matters:** Even if this baseline underperforms (which is likely given limited data), that is the most compelling evidence for the architectural thesis. If it somehow competes, that's also a meaningful finding worth reporting.

**Implementation notes:**
- Use Stable-Baselines3 for PPO — clean, well-tested, minimal custom code
- Gym environment wrapper around existing backtest engine
- Keep it simple: the point is a fair comparison, not a state-of-the-art end-to-end agent

### 3.2 Volatility-Targeting Baseline (Required — tests RQ2)

**What:** Scale the factor portfolio's invested fraction inversely proportional to recent realized or forecast volatility.

**Formula:**
```
invested_fraction_t = min(1.0, σ_target / σ_forecast_t)
```

Where `σ_target` is calibrated to match the full pipeline's average realized volatility (so the comparison is apples-to-apples on risk budget).

**Why it matters:** This is the simplest "smart" risk management rule. If RL can't beat this, the control layer's complexity isn't justified.

**Implementation:** ~30 lines on top of the existing factor-only strategy.

### 3.3 Drawdown-Based Deleveraging Baseline (Required — tests RQ2)

**What:** Cut exposure when trailing drawdown breaches thresholds, re-enter gradually.

**Design:**
```
if drawdown > -5%:  invested = 100%
if drawdown > -8%:  invested = 70%
if drawdown > -12%: invested = 40%
re-entry: increase by 10% per week when drawdown recovers above threshold
```

Thresholds calibrated to be reasonable, not optimized on the test set.

**Why it matters:** Tests whether the hedge RL layer is doing anything a simple drawdown rule can't.

### 3.4 Risk Parity / Min-Variance (Nice to have)

**What:** Inverse-variance weighting or full min-variance optimization using the same Ledoit-Wolf covariance estimator already in the pipeline.

**Why:** Reference point for "does the alpha layer matter, or is smart diversification enough?"

### 3.5 Updated Benchmark Summary

| Benchmark | Tests | Priority | Status | Effort |
|-----------|-------|----------|--------|--------|
| SPY buy-and-hold | Baseline | Already done | Implemented | — |
| Equal weight | Universe effect | Already done | Implemented | — |
| Factor-only | Alpha value | Already done | Implemented | — |
| Factor + constrained optimizer (no RL) | Optimizer value | Already done | Implemented | — |
| **Vol-targeting on factor portfolio** | RQ2 | **Required** | Implemented | Half day |
| **Drawdown-based deleveraging** | RQ2 | **Required** | Implemented | Half day |
| **End-to-end RL (PPO/DQN)** | RQ1 | **Required** | Implemented, expensive to run | 1–2 days |
| Risk parity / min-variance | Diversification reference | Nice to have | Partially implemented via min-var stabilizer / risk-parity path | Half day |

---

## 4. New Data & Features

### 4.1 Regime & Risk-State Enrichment (High Priority)

These directly improve the RL controller's information set and are cheap to add via FRED.

| Feature | Source | Role |
|---------|--------|------|
| VIX (VIXCLS) | FRED | Volatility regime state for RL |
| ICE BofA HY OAS (BAMLH0A0HYM2) | FRED | Credit stress indicator |
| 10Y–2Y term spread | Already have yields | Yield curve slope for macro regime |
| DXY / dollar index proxy | FRED or computed | Cross-asset risk-off signal |

**Implementation:** Add to `data.py` macro ingestion, include as RL state features. Respect the existing information-timing contract (apply configurable lag).

### 4.2 Uncertainty & Disagreement Features (High Priority — Distinctive)

These are the most intellectually distinctive additions and connect to the Bayesian RL coursework.

| Feature | Computation | Role |
|---------|-------------|------|
| Alpha-sleeve dispersion | Std dev of signal scores across sleeves for each asset | Measures model disagreement |
| Regime-belief entropy | −Σ p log p over HMM + macro regime posterior | Measures regime uncertainty |
| Rolling IC instability | Std dev of recent Spearman ICs per sleeve | Measures signal reliability |
| Forecast confidence interval width | From GARCH variance of variance | Measures vol forecast uncertainty |

**Why this matters:** Most RL-for-trading papers condition on point estimates. Conditioning on *uncertainty about those estimates* is a principled extension that connects to BA-MDPs and robust RL. This could be framed as a minor methodological contribution.

### 4.3 Point-in-Time Fundamentals (Medium Priority)

**Current state:** SEC quality signal is a static cross-sectional prior.

**Improvement:** Use SEC EDGAR filing dates to construct a proper point-in-time fundamentals panel. For each company, only use financial data that was publicly available as of the backtest date.

**Realistic scope:** Even a basic version (filing-date-aware trailing EPS, book value, ROE) would significantly improve credibility.

**Fallback:** If too time-consuming, keep the current static prior but add the strict-timing ablation as a formal robustness check (already partially implemented).

### 4.4 Data NOT to Prioritize

- News sentiment (requires serious NLP pipeline, dilutes focus)
- Social media data (noise, reproducibility issues)
- Alternative data (satellite, web scraping — interesting but off-topic)
- High-frequency data (different problem, different paper)
- Massively expanded universe (keep the current ~20 names, justify the choice)

---

## 5. Methodology Improvements

### 5.1 Reward Function Formalization & Ablation

**Problem:** Section 5.7 is the thinnest part of the paper relative to its importance. The reward design is a critical RL design choice and currently gets one paragraph with no equations.

**Fix:**
1. Write out the differential Sharpe reward formally:
   ```
   R_t = (r_t - μ̂_t) / (σ̂_t + ε)
   ```
   where μ̂ and σ̂ are exponentially weighted running estimates.

2. Write out the asymmetric hedge reward:
   ```
   R_hedge_t = r_t · 1(r_t ≥ 0) + λ_loss · r_t · 1(r_t < 0),  λ_loss > 1
   ```

3. **Run a reward ablation** comparing:
   - Differential Sharpe (current)
   - Raw return
   - Sortino-style (penalize downside vol)
   - Mean-variance style (return − λ · variance)

   Report how sensitive the results are to reward choice. If differential Sharpe is clearly best, that's a contribution. If results are robust to reward choice, that's also informative.

### 5.2 Expand Rolling Windows

**Current:** 2 rolling windows — not enough for distributional claims.

**Target:** 4–6 overlapping windows. Use shorter subperiods if necessary (e.g., 12–18 month test windows rolled every 6 months).

**Report:** Median and interquartile range of Sharpe, Calmar, max drawdown across windows. Box plots or violin plots instead of point estimates.

### 5.3 LSTM Sleeve — Decide Its Fate

**Options:**
- **Keep + ablate:** Run factor-only vs. factor+LSTM vs. full pipeline. If LSTM adds marginal alpha, document it.
- **Remove + note as future work:** If the adaptive combiner consistently downweights it, simplify the pipeline.

**Recommendation:** Run the ablation. If the LSTM adds < 0.5% return or < 0.05 Sharpe improvement on average, remove it and note that deep forecast sleeves did not improve the classical factor engine in this setting. That's an honest, useful finding.

### 5.4 Pairs Trading Sleeve — Same Decision

Check the adaptive combiner weights for the pairs sleeve. If it's consistently low-weighted, consider the same treatment as the LSTM.

### 5.5 Statistical Significance

**Current gap:** No formal significance testing.

**Options:**
- Block bootstrap on daily returns (preserve autocorrelation structure)
- Ledoit-Wolf Sharpe ratio test (comparison of Sharpe ratios accounting for estimation error)
- At minimum: report standard errors on Sharpe and Calmar estimates

This doesn't need to be heavy — even bootstrap confidence intervals on Sharpe would elevate the paper significantly.

**Status update:** Block-bootstrap confidence intervals and pairwise bootstrap
significance summaries are now implemented in the evaluation engine. What
remains is to promote those outputs into the main empirical narrative of the
paper once a full research run completes.

### 5.6 Additional Metrics

The checklist suggests a few additional metrics that are worth splitting into
``already available'' versus ``still worth adding.''

**Already available or effectively covered**
- Tail risk via the worst 5\% daily return / VaR-style reporting
- Downside deviation via the Sortino ratio

**Worth adding next**
- Explicit hedge cost versus hedge benefit summary
- Drawdown duration as a reported metric
- Possibly a more explicit downside semi-variance table in the paper

These are good additions because they strengthen the downside-control story
without changing the architecture.

### 5.7 Options Integration (Stretch RQ4)

This is the biggest realism upgrade still missing from the current codebase.

**Goal**
- Replace the stylized convex hedge sleeve with explicit option-based hedging

**Minimal strategy set**
- Protective put
- Collar
- Optional put spread

**RL extension**
- Let RL select both hedge type and hedge intensity

**Additional features needed**
- Implied volatility
- IV percentile
- Volatility regime / term structure context

**Status**
- Not implemented
- High-value future work
- Should be framed in the plan as a second-phase extension, not as a missing
  piece that invalidates the current paper

---

## 6. Paper Structure (Revised)

### Proposed outline for the August version:

1. **Introduction**
   - Motivation: RL in quantitative finance — controller vs. end-to-end
   - Research questions (RQ1, RQ2, RQ3)
   - Central defended statement
   - Contributions (5 items, revised to reflect new baselines)

2. **Related Work**
   - Factor investing literature (Fama-French, Jegadeesh-Titman)
   - RL for trading: end-to-end approaches (FinRL family, DeepPocket, etc.)
   - RL for execution (Almgren-Chriss, optimal execution literature)
   - RL as controller (distinguish from end-to-end — this is your positioning)
   - Rule-based risk management (vol targeting, trend-following overlays)
   - Regime detection and conditional allocation

3. **System Architecture**
   - High-level diagram (keep Figure 1)
   - Module decomposition (keep Table 1)
   - Information-timing contract
   - Constrained allocator design

4. **Methodology**
   - Alpha layer: factors, pairs, GARCH, HMM, (LSTM if kept), adaptive combiner
   - Control layer: portfolio RL, hedge RL, reward design (**expanded with equations**)
   - Execution: no-trade band, transaction costs
   - **New:** Uncertainty features as RL state inputs

5. **Experimental Design**
   - Universe and data sources
   - Walk-forward protocol
   - **Comparison 1:** Alpha stack viability (vs. SPY, equal weight)
   - **Comparison 2:** RL control value (ablation table, vs. rule-based baselines)
   - **Comparison 3:** Modular vs. end-to-end RL
   - Robustness: rolling windows, sensitivity sweeps, timing discipline

6. **Results**
   - Main result table (updated with new baselines)
   - Ablation results (with RQ3 framing)
   - Rule-based baseline comparison (RQ2)
   - End-to-end RL comparison (RQ1)
   - Rolling-window robustness
   - Regime-conditional behavior
   - Reward ablation
   - Statistical significance

7. **Discussion**
   - What the results mean for RL in quant finance
   - When RL control helps and when it doesn't
   - Limitations of the evaluation
   - Connection to broader sequential decision-making literature

8. **Threats to Validity**
   - Researcher degrees of freedom (prominent, honest treatment)
   - Universe selection sensitivity
   - Single historical sample
   - Stylized transaction costs and hedge payoffs
   - Data limitations (point-in-time, vintage)

9. **Conclusion**
   - Answers to RQ1–RQ3
   - Practical implications
   - Future work

10. **Appendix**
    - RL control ladders
    - Research artifacts list
    - Extended diagnostic figures (move most current figures here)
    - Hyperparameter tables

---

## 7. Related Work — Key References to Add

### End-to-end RL for trading (your "alternative approach")
- **FinRL** (Liu et al., 2020–2024) — the most prominent end-to-end RL trading framework
- **DeepTrader** (Wang et al., 2021) — RL for portfolio management
- **Moody & Saffell (2001)** — differential Sharpe ratio, direct RL for trading

### RL for execution
- **Almgren & Chriss (2000)** — already cited
- **Ning et al. (2021)** — deep RL for optimal execution

### Rule-based risk management
- **Moskowitz, Ooi, Pedersen (2012)** — time-series momentum and trend-following
- **Hocquard et al. (2013)** — volatility targeting
- **Harvey et al. (2018)** — managed portfolios, risk targeting

### Regime detection and conditional allocation
- **Ang & Bekaert (2002)** — regime switching and asset allocation
- **Guidolin & Timmermann (2007)** — asset allocation under multivariate regime switching

### Position: the gap you fill
Nobody has published a clean, ablated comparison of:
1. classical alpha + static allocation
2. classical alpha + rule-based risk control
3. classical alpha + RL control
4. end-to-end RL

...on the same universe, same features, same evaluation framework. That is a real gap in the literature.

---

## 8. Figures — Consolidation Plan

The current draft has 6 figures, several with 6–9 subplots each. For a published paper, this is too many in the main text.

### Main text figures (target: 5–6)

| Figure | Content | Purpose |
|--------|---------|---------|
| 1 | Architecture diagram | System overview (keep as-is) |
| 2 | Main equity curves: full pipeline, factor-only, vol-target, drawdown-rule, end-to-end RL, SPY | Core result visualization |
| 3 | Ablation bar chart: Sharpe and max drawdown across component configurations | RQ3 answer |
| 4 | Rolling-window box plots: Sharpe and Calmar distributions across windows | Robustness evidence |
| 5 | Regime-conditional controller behavior: invested fraction and hedge ratio by regime | Behavioral evidence |
| 6 | Reward ablation comparison | Sensitivity evidence |

### Appendix figures
- Alpha model diagnostics (current Figure 5)
- Execution RL details (current Figure 6)
- Full diagnostic dashboard (current Figure 4)
- Correlation matrices
- Per-sleeve adaptive weights

---

## 9. Writing & Positioning Notes

### Tone
- **Don't oversell.** The current draft is already good at this — keep it.
- **Be explicit about what you're NOT claiming.** Not production-ready, not novel in any single component, not exhaustively robust.
- **Frame limitations as methodology.** "The robustness engine tests X" rather than "we couldn't test Y."

### Universe justification
Add a paragraph explaining why these ~20 names. Acknowledge that results may depend on universe selection (e.g., NVDA's exceptional run). Suggest universe sensitivity as future work.

### Researcher degrees of freedom
Dedicate a full paragraph in Threats to Validity. Acknowledge that the architecture was iterated by observing backtest behavior. Explain how the ablation framework partially mitigates this (since component contributions are isolated) and where it doesn't (the overall design was still informed by results).

### ResearchGate-specific notes
- Include a clear abstract with quantitative results
- Tag with relevant keywords: reinforcement learning, quantitative trading, portfolio management, factor investing, risk management
- Link the GitHub repo prominently
- Consider a 2–3 sentence "plain language summary" for broader reach


---

## 10. Open Decisions

These are choices that need to be made during implementation, not upfront:

1. **End-to-end RL algorithm:** PPO vs. DQN vs. SAC. Recommendation: PPO (continuous action space, well-tested in SB3). Consider also running DQN with discrete buckets for a second comparison.

2. **LSTM sleeve:** Keep or cut? Run the ablation first, decide based on marginal contribution.

3. **Pairs sleeve:** Same as above.

4. **Point-in-time fundamentals:** Full SEC EDGAR filing-date pipeline or keep as static prior with strict-timing ablation? Depends on time budget — if April baselines take longer than expected, defer this.

5. **Statistical significance method:** Block bootstrap vs. Ledoit-Wolf Sharpe test. Recommendation: block bootstrap (more general, handles all metrics).

6. **Paper length target:** Current draft is ~19 pages. Revised version will likely be 22–28 pages. That's fine for a self-published technical report / working paper on ResearchGate.

7. **Single author vs. acknowledgments:** If anyone (advisor, colleague) gives substantial feedback, acknowledge them. Keep single-author unless there's a co-author situation.

---

## 11. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| End-to-end RL baseline performs surprisingly well | Weakens RQ1 claim | Report honestly; reframe as "both approaches viable, modular more interpretable" |
| Rule-based baselines match RL on risk metrics | Weakens RQ2 claim | Report honestly; argue RL is more adaptive across regimes (show regime-conditional evidence) |
| Expanded rolling windows show high variance | Weakens robustness story | Report distributions, not just medians; discuss sample limitations honestly |
| LSTM/pairs add nothing | Simplify pipeline | Cut and report as negative finding — this is publishable and honest |
| Point-in-time fundamentals too time-consuming | Credibility gap | Keep strict-timing ablation, acknowledge limitation explicitly |
| August deadline too tight | Incomplete paper | Prioritize Tier 1 items; publish with "working paper" label and iterate |

---

## 12. Success Criteria

The paper is "worth publishing" if it delivers:

1. ✅ A clear, testable research question that someone else would care about
2. ✅ At least one comparison that doesn't exist in the current literature (modular RL vs. end-to-end RL vs. rule-based, same evaluation framework)
3. ✅ Honest evaluation with ablations, multiple windows, sensitivity analysis
4. ✅ Reproducible via public GitHub repo
5. ✅ Results that survive basic robustness checks (even if imperfect)
6. ✅ Writing that is precise, non-promotional, and self-aware about limitations
