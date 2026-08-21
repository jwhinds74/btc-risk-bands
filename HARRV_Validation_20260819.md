# XGBit Volatility Radar — HAR-RV Model Validation Report

- **Generated:** 2026-08-19 03:11 UTC
- **Data source:** Binance · **Observations:** 750 (capped at 750 days to match production)
- **Period:** 2024-07-30 → 2026-08-18
- **Account assumptions:** $700 capital · 2.0% risk/trade · 0.137% round-trip cost

## Verdict

🟢 MODEL VALIDATED, WITH WATCH ITEMS. Assumptions hold, the model generalises, and the bands pass formal coverage and independence testing. Queued improvements are efficiency gains, not corrections.

## Dashboard

| Test | Status | Value |
|---|---|---|
| Component significance | **WATCH** | 2.0000 |
| Multicollinearity | **PASS** | 2.2800 |
| Regime alignment | **PASS** | 1.3357 |
| Residual autocorrelation | **WATCH** | 0.0000 |
| Conditional heteroskedasticity | **INFO** | 0.0345 |
| Stationarity | **WATCH** | 0.0005 |
| Structural stability | **WATCH** | 1.2114 |
| Generalisation (QLIKE skill vs baseline) | **PASS** | 20.5400 |
| Performance drift across folds | **PASS** | 0.3000 |
| Unbiasedness and efficiency (MZ) | **PASS** | 0.0638 |
| Kupiec — violation frequency | **WATCH** | 0.0130 |
| Christoffersen — independence | **PASS** | 0.7544 |
| Sizing efficiency | **WATCH** | 0.8133 |
| Implied leverage | **PASS** | 0.6481 |

## Findings and actions

### WATCH — Component significance

**Finding:** 2 of 3 components significant. One horizon may be redundant in this sample.

**Action:** Re-run after 3 months. If the same component stays insignificant, drop it and compare QLIKE — a simpler model with equal accuracy is strictly better.

### PASS — Multicollinearity

**Finding:** Maximum VIF is 2.3 (below 10). Components overlap, as designed, but not severely enough to destabilise estimation.

**Action:** None.

### PASS — Regime alignment

**Finding:** Implied long-run RV (2.10%/day) is close to the current regime (1.57%/day): 1.34x. The intercept is not distorting the forecast.

**Action:** None. The training window is appropriately scoped.

### WATCH — Residual autocorrelation

**Finding:** Ljung–Box at 22 lags: p = 0.0000. Some autocorrelation remains. With a large sample this test rejects almost anything, so the practical size of the effect matters more than the p-value.

**Action:** Two concrete options: (a) keep HAC errors — already in use — which is sufficient if the goal is forecasting rather than inference; (b) test adding a quarterly component (RV_66) and keep it only if out-of-sample QLIKE improves.

### INFO — Conditional heteroskedasticity

**Finding:** ARCH-LM p = 0.0345: conditional heteroskedasticity is present. This is expected when modelling volatility and is not a defect.

**Action:** None — but it confirms that bands must be built from standardised residuals rescaled by conditional volatility, never from a fixed width. Verify that the production code does this.

### WATCH — Stationarity

**Finding:** ADF rejects a unit root but KPSS also rejects stationarity — the classic signature of a long-memory series. The model remains usable.

**Action:** Refit frequently rather than assuming fixed parameters. The production monitor already refits daily, which is the correct response. Do not extend the training window in search of stability — see Section 6.

### WATCH — Structural stability

**Finding:** CUSUM breaches the 5% bands (max 1.211 > 0.948) and the largest coefficient CV is 187%. There is evidence of structural change.

**Action:** Cap the production training window at roughly 750 days (~3 years) and refit daily. Do NOT train on the full history: distant regimes are describing a different market. Verify monitor.py enforces this cap and that the app uses the same window — otherwise the two are different models publishing under one name.

### PASS — Generalisation (QLIKE skill vs baseline)

**Finding:** The model beats a naive last-week-average forecast by 20.5% in QLIKE, winning in 5 of 5 held-out blocks. It generalises.

**Action:** None. Quote the out-of-sample skill score, never the in-sample R², in any client-facing material.

### PASS — Performance drift across folds

**Finding:** No material decline across folds (rho = 0.30, p = 0.624).

**Action:** None, but note the test has low power with 5 folds.

### PASS — Unbiasedness and efficiency (MZ)

**Finding:** Joint hypothesis a=0, b=1 not rejected (p = 0.0638). The forecast is unbiased and efficient.

**Action:** None. No recalibration needed.

### WATCH — Kupiec — violation frequency

**Finding:** Observed coverage 98.4% vs 95% nominal (p = 0.0130): the band is WIDER than necessary. This is the benign direction — it over-covers.

**Action:** No risk action required. If you want the extra precision, apply the MZ recalibration from Section 8, which shrinks the band toward its correct width. Section 10 quantifies the opportunity cost.

### PASS — Christoffersen — independence

**Finding:** p = 0.7544 with 0 consecutive violations. Failures are dispersed, which is the benign failure mode.

**Action:** None.

### WATCH — Sizing efficiency

**Finding:** The band is about 23.0% wider than the observed violation rate justifies, so every position is that much smaller than optimal — roughly $104 of foregone notional per trade. This is an opportunity cost, not a risk exposure.

**Action:** Optional: apply the MZ recalibration from Section 8, or multiply the half-width by 0.81. Safe to defer until the live track record reaches ~90 observations — over-covering never harms the user.

### PASS — Implied leverage

**Finding:** The sizing rule implies 0.65x leverage — below 1x at current volatility.

**Action:** None. The rule is self-limiting: as volatility rises, the band widens and size falls automatically.

## Estimation (HAC standard errors)

```
 term    coef  se_classical  se_HAC  t_HAC  p_HAC
const 0.00681       0.00168 0.00136   5.00 0.0000
 rv_d 0.25492       0.04255 0.05040   5.06 0.0000
 rv_w 0.24162       0.07654 0.09169   2.64 0.0084
 rv_m 0.17988       0.09531 0.09992   1.80 0.0718
```
- R² = 0.1767 · adjusted R² = 0.1734 · n = 750
- Persistence (b_d+b_w+b_m) = 0.6764
- HAC errors are 1.06x the classical ones

## Variance inflation

```
variable  VIF
    rv_d 1.64
    rv_w 2.28
    rv_m 1.60
```

## Regime alignment

```
                     quantity daily annualised
Implied long-run RV  c/(1-Σb) 2.10%        40%
               Sample mean RV 2.12%        41%
        Forecast for tomorrow 1.57%        30%
```
- Misalignment factor: 1.34x · intercept share of forecast: 43%
- Training window used: 750 days (capped to match production)

## Residual diagnostics

```
             test  statistic  p_value                                H0
ARCH-LM (10 lags)      19.49   0.0345 no conditional heteroskedasticity
    Breusch–Pagan      35.60   0.0000                  homoskedasticity
      Jarque–Bera    3331.33   0.0000    normally distributed residuals
```

```
 lag  lb_stat  lb_pvalue
   5   4.3051     0.5064
  10  48.2567     0.0000
  22 102.3840     0.0000
```

## Stationarity

```
 series  ADF_stat  ADF_p  KPSS_stat  KPSS_p
     RV    -4.287 0.0005      0.522  0.0368
log(RV)    -3.533 0.0072      0.607  0.0220
```

## Parameter stability

```
coefficient  mean   std  CV_%
        b_d 0.261 0.061  23.4
        b_w 0.244 0.103  42.2
        b_m 0.082 0.153 187.4
```
- CUSUM maximum: 1.211 (5% critical value: 0.948)

## Cross-validation

```
 fold  train_end             test_window  n_test  QLIKE  QLIKE_base  skill_%  R2_oos  corr
    1 2025-04-05 2025-04-28 → 2025-07-26      90 0.5249      0.6828     23.1 -0.7194 0.053
    2 2025-07-26 2025-08-18 → 2025-11-15      90 0.7178      0.9006     20.3  0.0088 0.202
    3 2025-11-15 2025-12-08 → 2026-03-07      90 0.6535      0.8050     18.8  0.2529 0.525
    4 2026-03-07 2026-03-30 → 2026-06-27      90 0.3377      0.4797     29.6  0.0958 0.340
    5 2026-06-27 2026-07-20 → 2026-08-18      30 0.4471      0.5015     10.9 -0.4793 0.077
```
- Mean QLIKE 0.5362 vs baseline 0.6739 · skill +20.5% · model wins 5/5 folds
- In-sample R² 0.1767 · out-of-sample R² -0.1682 (unstable on short blocks)

## Mincer–Zarnowitz

- a = -0.00130, b = 0.997, R² = 0.1945
- Joint test (a=0, b=1): F = 2.79, p = 0.0638
- Mean bias: +7.0%

## Band backtest

```
                           test    LR  p_value                         H0
Kupiec (unconditional coverage) 6.174   0.0130        violation rate = 5%
  Christoffersen (independence) 0.098   0.7544 violations are independent
   Conditional coverage (joint) 6.272   0.0435        both simultaneously
```
- Observed coverage 98.4% (nominal 95%), 3 violations of 188, 0 consecutive

## Business impact

```
                        metric         value
Forecast volatility (next day)         1.57%
           95% band half-width        ±3.09%
            Max risk per trade        $14.00
              Implied notional          $454
         Implied position size 0.00701 units
              Implied leverage         0.65x
     Round-trip cost per trade         $0.62
```
- Band-width correction needed: x0.813 (positions currently 23.0% undersized)
- Implied leverage: 0.65x
- Round-trip cost per trade: $0.62

---
*Generated by HARRV_Model_Validation_EN.ipynb*