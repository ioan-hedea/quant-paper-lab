# Architecture Revision v2: Factor-Anchored Portfolio Control
## From RL-Centric to Control-Method Comparison

---

## 1. Motivation for the Revision

The March 28, 2026 frozen-bundle evaluation exposed four structural weaknesses:

1. **Hedge RL destroys value.** Adding hedge RL drops Sharpe from 0.79 (alpha + portfolio RL) to 0.43 (full pipeline). The regime-conditioned behavior is inverted: more hedging in bull states, less in bear states.
2. **Rich state features hurt.** Removing regime, uncertainty, or volatility inputs improves the full pipeline in every case. The tabular Q-learner cannot exploit them.
3. **Pairs and LSTM add nothing.** Both sleeves slightly worsen drawdown and add no incremental Sharpe over the factor core.
4. **RL does not beat simple rules.** Vol-target (Sharpe 0.89) and DD-delever (0.92) both dominate the full RL pipeline (0.43) decisively.

The revision therefore pivots from "does RL work as a controller?" to "what is the best control mechanism on top of a strong factor engine?" This makes every empirical outcome publishable.

---

## 2. Revised Research Questions

### RQ1: Which control mechanism adds the most value?

Given a strong finance-first alpha engine, compare:
- Fixed allocator (no learning)
- Rule-based overlays (vol-target, DD-delever, regime rules, ensemble)
- Contextual bandit (LinUCB / Thompson Sampling)
- CVaR-aware robust optimization
- Reinforcement learning (tabular Q-learning, PPO)

Evaluation on Sharpe, Calmar, max drawdown, and bootstrap significance.

### RQ2: Can a simple control layer improve downside without degrading Sharpe?

Test whether any control mechanism improves Calmar and max drawdown relative to the factor-only benchmark without losing more than 0.10 Sharpe. This is the practitioner question.

### RQ3: Which sources of complexity are justified?

Ablation-driven assessment of:
- Richer alpha sleeves (pairs, LSTM) vs. factor-only
- Richer state inputs (regime, uncertainty, volatility) vs. minimal state
- Learned hedge logic vs. no hedge
- Learned control vs. rule-based control vs. optimization-based control

---

## 3. Retained Components (Proven Value)

### 3.1 Alpha Layer

| Component | Role | Justification |
|-----------|------|---------------|
| Cross-sectional factor model | Composite alpha score (momentum, value, quality, low-vol) | Factor-only benchmark is the strongest alpha engine in the archived bundle (Sharpe 1.00 in main split, 0.82 averaged across splits) |
| GARCH(1,1) | Per-asset conditional volatility forecast | Used for confidence scaling and risk-awareness; stable contribution |
| HMM (2-state) | Bull/bear regime belief | Blended with macro belief; provides regime context to control layer |
| Adaptive combiner | IC-weighted signal combination | With fewer sleeves (factor + GARCH + HMM only), consider whether adaptive weighting still adds value vs. fixed weights — run ablation |

### 3.2 Portfolio Construction

| Component | Role | Justification |
|-----------|------|---------------|
| Constrained intermediate allocator | Long-only, capped, turnover-penalized QP | Already implements institutional-grade portfolio constraints; foundation for CVaR extension |
| Target book: 90% factor + 10% stabilizer | Factor-anchored allocation | Preserves interpretability; stabilizer uses Ledoit-Wolf shrinkage covariance |
| No-trade band | Turnover suppression | Economically meaningful with the 3-part cost model |
| 3-part transaction cost model | Realistic execution drag | Fixed bps + vol-scaled + concentration penalty |

### 3.3 Evaluation Infrastructure

| Component | Role |
|-----------|------|
| Frozen-bundle protocol | Reproducibility and anti-snooping discipline |
| Walk-forward evaluation | Out-of-sample daily roll |
| Block-bootstrap significance | Confidence intervals preserving serial dependence |
| Jobson-Korkie tests | Parametric Sharpe comparisons |
| Rolling-window robustness | Internal stability assessment |
| Blocked time-series CV | Cross-validation respecting temporal ordering |
| Regime-conditioned diagnostics | Behavioral analysis of controller actions |

---

## 4. Removed Components (Ablation-Justified)

| Component | Reason for Removal | Evidence |
|-----------|-------------------|----------|
| Pairs (cointegration) sleeve | No incremental Sharpe; slightly worsens drawdown | Table 4: Factor+pairs Sharpe 0.80 vs. Factor-only 0.82 |
| LSTM forecast sleeve | No incremental Sharpe; slightly worsens drawdown | Table 4: Factor+LSTM Sharpe 0.81 vs. Factor-only 0.82 |
| Hedge RL agent | Main driver of Sharpe collapse; inverted regime behavior | Table 4: Alpha+hedge RL Sharpe 0.49 vs. Alpha+portfolio RL 0.79 |
| Regime state input | Removing it improves full pipeline | Table 4: Full (no regime) Sharpe 0.47 > Full 0.43 |
| Uncertainty state input | Removing it improves full pipeline | Table 4: Full (no uncert.) Sharpe 0.44 > Full 0.43 |
| Volatility state input | Removing it improves full pipeline | Table 4: Full (no vol) Sharpe 0.46 > Full 0.43 |
| Synthetic options sleeve (as core) | Proxy payoffs not market-priced; hedge agent can't use them effectively | Hedge RL with IV-conditioned overlays still underperforms no-hedge variant |

---

## 5. New Control Candidates

### Candidate A: Rule-Based Control Suite (Baseline)

**Purpose:** Establish the performance ceiling for non-learned control. Every other candidate must beat this.

#### A1: Fixed Allocator
- Constant invested fraction (e.g., 95%)
- No overlay, no dynamic adjustment
- Pure factor-alpha + constrained allocator

#### A2: Volatility Targeting
- Target portfolio volatility σ_target (e.g., 12%)
- Invested fraction: b_t = min(1, σ_target / σ̂_t)
- σ̂_t from trailing realized vol (e.g., 63-day)
- Already implemented in current codebase

#### A3: Drawdown-Based Deleveraging
- When drawdown exceeds threshold d_thresh: reduce exposure linearly
- b_t = 1 if DD_t > -d_thresh, else max(b_min, 1 - λ_dd * |DD_t|)
- Already implemented in current codebase

#### A4: Regime-Conditioned Exposure Rules
- Use HMM regime belief directly:
  - Bull (belief > 0.7): b_t = 1.0
  - Neutral (0.3 < belief < 0.7): b_t = 0.90
  - Bear (belief < 0.3): b_t = 0.75
- Thresholds and exposure levels are hyperparameters — keep them few and coarse

#### A5: Simple Ensemble
- b_t = mean(b_t^{vol-target}, b_t^{dd-delever}, b_t^{regime})
- Or: b_t = min(b_t^{vol-target}, b_t^{dd-delever}, b_t^{regime})
- Tests whether combining simple rules is better than any single rule

**Implementation effort:** Low. A2 and A3 exist. A1 is trivial. A4 and A5 are ~50 lines each.

---

### Candidate B: Contextual Bandit

**Purpose:** Test whether a lightweight learned controller can beat rules without the complexity of full RL.

#### Why it fits this problem:
- Action space is small and discrete (5 invested-fraction buckets)
- Temporal credit assignment across days is noisy and may not help
- The decision is essentially: "given current state, which exposure bucket maximizes risk-adjusted return?"
- No need to model state transitions or long-horizon returns

#### Specification:

**State vector (kept minimal):**
- Alpha strength: mean absolute alpha score across universe
- Recent drawdown: trailing max drawdown over 63 days
- Recent realized vol: trailing 21-day realized portfolio volatility

**Action space:**
- 5 invested-fraction buckets: {82%, 90%, 95%, 98%, 100%}
- Optionally: 3 overlay-size buckets: {5%, 10%, 15%}

**Reward:**
- Sortino-style: r_t / σ̂_downside with asymmetric loss penalty
- Computed over a short forward window (5–10 days) to reduce noise

**Algorithms to compare:**
1. **LinUCB** — linear contextual bandit with upper confidence bound exploration
2. **Thompson Sampling** — Bayesian posterior sampling over arm rewards
3. **ε-greedy with linear model** — simplest baseline

**Key design decisions:**
- Window for reward computation: 5 days (weekly Sortino) vs. 10 days vs. 21 days — ablate
- Exploration parameter tuning: UCB α or ε schedule
- Feature normalization: rolling z-score over trailing 252 days

**Implementation effort:** Medium. ~200-300 lines for the bandit framework. Use existing state construction, just with fewer features.

---

### Candidate C: Supervised Regime-Conditioned Controller

**Purpose:** Test whether a simple classifier can learn the optimal exposure mapping without online exploration.

#### Approach:
1. **Offline labeling:** For each historical date, compute which invested-fraction bucket would have maximized trailing 21-day Sortino ratio
2. **Train classifier:** Map (alpha_strength, recent_drawdown, recent_vol) → best exposure bucket
3. **Models:** Logistic regression, small random forest, or single decision tree
4. **Walk-forward:** Retrain on expanding window, predict next day's exposure

#### Why include it:
- It's the "supervised learning ceiling" — if a simple classifier can't find the mapping, the bandit won't either
- Decision tree variant is fully interpretable (can print the rules)
- If it works well, it suggests the problem doesn't need exploration at all

**Implementation effort:** Low-medium. ~150 lines. Scikit-learn handles the models.

---

### Candidate D: CVaR-Aware Robust Optimization

**Purpose:** Replace learned control with optimization-based control that has formal risk guarantees.

#### Approach:

Extend the existing constrained allocator by adding a CVaR penalty to the objective:

```
min_w   λ_risk * w'Σ_t w  -  λ_alpha * α_t'w  +  λ_anchor * ||w - w_target||²
        + λ_turn * ||w - w_{t-1}||²  +  λ_cvar * CVaR_γ(w, Σ_t)

s.t.    w_i ≥ 0,  Σ_i w_i = 1,  per-asset caps,  group limits
```

Where CVaR_γ is the conditional value-at-risk at confidence level γ (e.g., 95%).

#### State-dependent risk budget:
- Use regime belief to modulate λ_cvar:
  - Bull: λ_cvar = λ_base
  - Neutral: λ_cvar = 2 * λ_base
  - Bear: λ_cvar = 4 * λ_base
- This gives regime-conditioned risk control without any learning

#### Drawdown budget extension:
- Add a soft constraint: if trailing drawdown > d_thresh, increase λ_risk and λ_cvar multiplicatively
- This embeds DD-delever logic directly into the optimizer

#### Why this is strong:
- Mathematically principled — CVaR is a coherent risk measure
- Convex formulation — solvable reliably with cvxpy
- Interpretable — every λ has a clear economic meaning
- Connects to robust RL literature (uncertainty sets, worst-case optimization)
- Natural extension of existing allocator code

**Implementation effort:** Medium. ~200-300 lines. Requires cvxpy. The hard part is getting the CVaR linearization right (use the Rockafellar-Uryasev formulation).

#### CVaR computation (Rockafellar-Uryasev):

For a portfolio w with return scenarios {r_1, ..., r_S} drawn from the estimated return distribution:

```
CVaR_γ(w) = min_ζ { ζ + 1/(S(1-γ)) * Σ_s max(0, -w'r_s - ζ) }
```

This is a linear program when embedded in the portfolio optimization.

Scenario generation: use the Ledoit-Wolf covariance matrix Σ_t and alpha scores α_t to sample S = 500-1000 return scenarios from N(α_t, Σ_t), then compute the CVaR penalty.

---

## 6. Retained RL Comparators

Keep these as comparators, not as the headline system:

### Tabular Q-Learning (Portfolio RL Only)
- State: (alpha_strength_bucket, recent_drawdown_bucket, recent_vol_bucket)
- Action: invested fraction bucket (5 levels)
- Reward: Sortino-style
- This is the "Alpha + portfolio RL" row from Table 4 (Sharpe 0.79)

### End-to-End PPO
- Same as current implementation
- Receives full state information, outputs allocation directly
- Serves as the "what if we gave everything to a single policy?" comparator

---

## 7. Updated Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Alpha & Signal Layer                   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Factor   │  │  GARCH   │  │   HMM    │               │
│  │ (4-style) │  │ (vol)    │  │ (regime) │               │
│  └─────┬─────┘  └─────┬────┘  └─────┬────┘               │
│        │              │             │                    │
│        └──────────┬───┘─────────────┘                    │
│                   │                                      │
│          ┌────────▼────────┐                             │
│          │ Adaptive / Fixed │                             │
│          │    Combiner      │                             │
│          └────────┬─────────┘                             │
│                   │  α_t, c_t                            │
└───────────────────┼──────────────────────────────────────┘
                    │
┌───────────────────┼──────────────────────────────────────┐
│                   │    Control Layer (COMPARED)           │
│                   ▼                                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Constrained Allocator (shared foundation)          │ │
│  │  Long-only, capped, turnover-penalized              │ │
│  │  Target: 90% factor + 10% stabilizer                │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│    ┌────────┬───────────┼───────────┬──────────┐         │
│    ▼        ▼           ▼           ▼          ▼         │
│  ┌────┐  ┌──────┐  ┌────────┐  ┌───────┐  ┌───────┐    │
│  │ A  │  │  B   │  │   C    │  │   D   │  │  RL   │    │
│  │Rule│  │Bandit│  │Superv. │  │ CVaR  │  │ Q/PPO │    │
│  │Base│  │LinUCB│  │Classif.│  │Robust │  │Retain │    │
│  └──┬─┘  └──┬───┘  └───┬────┘  └───┬───┘  └───┬───┘    │
│     │       │          │           │          │         │
│     └───────┴──────────┴───────────┴──────────┘         │
│                         │                                │
│                  b_t (invested fraction)                  │
│                  τ_t (overlay size, if applicable)        │
│                                                          │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────┼──────────────────────────────────────┐
│                   ▼    Execution & Evaluation             │
│  ┌──────────────────────────────────────┐                │
│  │  No-Trade Band → Cost Model → PnL   │                │
│  └──────────────────────────────────────┘                │
│  ┌──────────────────────────────────────┐                │
│  │  Frozen-Bundle Evaluation Protocol   │                │
│  │  • Walk-forward  • Bootstrap CI      │                │
│  │  • Ablation      • Rolling windows   │                │
│  │  • Regime diagnostics • Jobson-Korkie│                │
│  └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

---

## 8. Evaluation Protocol

### 8.1 Primary Comparison Table

All candidates evaluated on the same frozen bundle, same walk-forward window, same cost model:

| Strategy | Return | Vol | Sharpe | Max DD | Calmar |
|----------|--------|-----|--------|--------|--------|
| Factor-only (no control) | — | — | — | — | — |
| SPY (passive) | — | — | — | — | — |
| A1: Fixed allocator | — | — | — | — | — |
| A2: Vol-target | — | — | — | — | — |
| A3: DD-delever | — | — | — | — | — |
| A4: Regime rules | — | — | — | — | — |
| A5: Rule ensemble | — | — | — | — | — |
| B: Contextual bandit (LinUCB) | — | — | — | — | — |
| B: Contextual bandit (Thompson) | — | — | — | — | — |
| C: Supervised controller | — | — | — | — | — |
| D: CVaR-robust optimizer | — | — | — | — | — |
| RL: Tabular Q (portfolio only) | — | — | — | — | — |
| RL: End-to-end PPO | — | — | — | — | — |

### 8.2 Statistical Tests

- Block-bootstrap Sharpe deltas for all pairwise comparisons
- Jobson-Korkie parametric tests
- Bootstrap CIs on Calmar and max drawdown differences
- Multiple-testing correction (Holm-Bonferroni) given the expanded comparison set

### 8.3 Ablation Design

1. Alpha ablation: factor-only vs. factor+GARCH vs. factor+GARCH+HMM
2. Combiner ablation: adaptive IC-weighted vs. fixed weights
3. Control-complexity ablation: no control → rules → bandit → CVaR → RL
4. State-space ablation (for bandit and RL): minimal vs. +regime vs. +vol vs. +uncertainty

### 8.4 Robustness

- Rolling 2-year windows (same protocol as current paper)
- Blocked time-series CV (5 folds)
- Regime-conditioned performance breakdown for each control method

---

## 9. Implementation Plan

### Phase 1: Foundation (Week 1)

- [ ] Strip pairs, LSTM, hedge RL, and rich state features from the pipeline
- [ ] Verify factor-only and alpha-stack-no-RL baselines reproduce Table 4
- [ ] Confirm existing vol-target and DD-delever baselines still work
- [ ] Implement A1 (fixed allocator), A4 (regime rules), A5 (ensemble)
- [ ] Run full evaluation suite on Candidates A1–A5
- [ ] **Checkpoint:** if the rule-based suite already shows a clear winner with Calmar improvement and Sharpe within 0.10 of factor-only, that is already a result

### Phase 2: Contextual Bandit (Week 2)

- [ ] Implement bandit framework: state construction, action selection, reward computation
- [ ] Implement LinUCB with 3-feature state, 5 exposure actions, Sortino reward
- [ ] Implement Thompson Sampling variant
- [ ] Implement ε-greedy baseline
- [ ] Ablate reward window (5 / 10 / 21 days)
- [ ] Run evaluation suite, compare to Phase 1 baselines

### Phase 3: CVaR-Robust Optimizer (Week 2–3)

- [ ] Extend constrained allocator with CVaR penalty (Rockafellar-Uryasev)
- [ ] Implement scenario generation from Ledoit-Wolf covariance
- [ ] Implement state-dependent λ_cvar modulation via regime belief
- [ ] Implement drawdown-budget soft constraint
- [ ] Tune λ_cvar, γ, and scenario count on training window
- [ ] Run evaluation suite, compare to Phase 1–2

### Phase 4: Supervised Controller & Comparators (Week 3)

- [ ] Implement offline labeling (best exposure bucket per date)
- [ ] Train logistic regression, random forest, decision tree
- [ ] Walk-forward expanding-window evaluation
- [ ] Re-run tabular Q-learning (portfolio only, Sortino reward, minimal state)
- [ ] Re-run PPO baseline
- [ ] Full comparison table with all candidates

### Phase 5: Paper Revision (Week 4)

- [ ] Freeze new archived bundle
- [ ] Generate all tables, figures, bootstrap CIs
- [ ] Rewrite Sections 1, 2, 3, 5, 6, 8, 9, 11 to reflect new RQs and architecture
- [ ] Update abstract and conclusion
- [ ] Archive on arXiv / ResearchGate

---

## 10. Changes to Paper Sections

| Section | Change |
|---------|--------|
| **Title** | Broader: e.g., "Control Mechanisms for Factor-Anchored Portfolio Construction: From Rules to Reinforcement Learning" |
| **Abstract** | Reframe: the paper compares control mechanisms (rules, bandits, robust optimization, RL) over a finance-first alpha engine. RL is one candidate, not the protagonist. |
| **§1 Introduction** | Rewrite RQs per Section 2 above. Motivation shifts from "is RL useful?" to "what is the right control complexity?" |
| **§2 Related Work** | Add contextual bandit literature (Li et al. 2010 LinUCB, Agrawal & Goyal 2013 Thompson Sampling), CVaR optimization (Rockafellar & Uryasev 2000), robust portfolio optimization (Goldfarb & Iyengar 2003). Keep existing RL and factor references. |
| **§3 Architecture** | Update Figure 1 to show parallel control candidates. Remove hedge RL, pairs, LSTM from diagram. |
| **§4 Data** | Mostly unchanged. Remove pairs-specific data discussion. |
| **§5 Methodology** | Remove §5.3 (Pairs), simplify §5.5 (combiner with fewer sleeves). Remove §5.8 hedging section. Add new subsections for contextual bandit, supervised controller, and CVaR optimizer. Keep §5.6 (Portfolio RL) as one candidate. |
| **§6 Experimental Design** | Expand benchmark set to include all candidates A–D. Describe the phase structure. |
| **§7 Metrics** | Add multiple-testing correction. Otherwise unchanged. |
| **§8 Results** | Restructure around new RQs. Main table becomes the full comparison. Ablation focuses on control-complexity ladder and alpha-sleeve pruning. |
| **§9 Discussion** | Reframe around "which control complexity is justified?" rather than "RL didn't work." |
| **§10 Threats** | Update for new candidates. Add bandit/CVaR specific caveats. |
| **§11 Conclusion** | Rewrite to reflect empirical ranking of control methods. |
| **Appendix** | Update action ladders. Add bandit hyperparameters, CVaR λ schedules. |

---

## 11. Expected Outcome Scenarios

### Best case
CVaR-robust optimizer or contextual bandit improves Calmar significantly over factor-only while staying within 0.05 Sharpe. Paper story: "lightweight, interpretable control adds real downside value; RL is unnecessary complexity for this problem." Strong, clean, publishable.

### Likely case
Rule-based ensemble and CVaR optimizer are competitive. Bandit is slightly better or tied. RL is in the middle. Paper story: "control complexity beyond simple rules is marginally justified, but the gains are small and context-dependent." Still publishable — the comparison methodology is the contribution.

### Worst case
Nothing beats factor-only on risk-adjusted terms after costs. Paper story: "a strong factor engine does not benefit from overlay control in this setting; the alpha is the entire story." This is still a useful and publishable negative result because it answers a question practitioners actually have.

In all three scenarios, the paper has a clear finding. That is the advantage of the revised RQ structure.
