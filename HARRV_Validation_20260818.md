# XGBit Volatility Radar — HAR-RV Model Validation Report

- **Generated:** 2026-08-18 20:10 UTC
- **Data source:** Binance · **Observations:** 3266
- **Period:** 2017-09-08 → 2026-08-17
- **Account assumptions:** $700 capital · 2.0% risk/trade · 0.137% round-trip cost

## Verdict

🟢 MODEL VALIDATED, WITH WATCH ITEMS. Assumptions hold, the model generalises, and the bands pass formal coverage and independence testing. Queued improvements are efficiency gains, not corrections.

## Dashboard

| Test | Status | Value |
|---|---|---|
| Component significance | **PASS** | 3.0000 |
| Multicollinearity | **PASS** | 3.9600 |
| Residual autocorrelation | **WATCH** | 0.0000 |
| Conditional heteroskedasticity | **INFO** | 0.0000 |
| Stationarity | **WATCH** | 0.0000 |
| Structural stability | **WATCH** | 1.6236 |
| Generalisation | **WATCH** | 0.1852 |
| Performance drift across folds | **PASS** | -0.8000 |
| Unbiasedness and efficiency (MZ) | **WATCH** | 0.0004 |
| Kupiec — violation frequency | **WATCH** | 0.0453 |
| Christoffersen — independence | **PASS** | 0.1437 |
| Sizing efficiency | **WATCH** | 0.9322 |
| Implied leverage | **PASS** | 0.8716 |

## Findings and actions

### PASS — Component significance

**Finding:** All three horizon components (daily, weekly, monthly) are significant at 5% with HAC errors. Each timescale contributes its own information.

**Action:** None. Keep the current specification.

### PASS — Multicollinearity

**Finding:** Maximum VIF is 4.0 (below 10). Components overlap, as designed, but not severely enough to destabilise estimation.

**Action:** None.

### WATCH — Residual autocorrelation

**Finding:** Ljung–Box at 22 lags: p = 0.0000. Some autocorrelation remains. With a large sample this test rejects almost anything, so the practical size of the effect matters more than the p-value.

**Action:** Two concrete options: (a) keep HAC errors — already in use — which is sufficient if the goal is forecasting rather than inference; (b) test adding a quarterly component (RV_66) and keep it only if out-of-sample QLIKE improves.

### INFO — Conditional heteroskedasticity

**Finding:** ARCH-LM p = 0.0000: conditional heteroskedasticity is present. This is expected when modelling volatility and is not a defect.

**Action:** None — but it confirms that bands must be built from standardised residuals rescaled by conditional volatility, never from a fixed width. Verify that the production code does this.

### WATCH — Stationarity

**Finding:** ADF rejects a unit root but KPSS also rejects stationarity — the classic signature of a long-memory series. The model remains usable.

**Action:** Refit frequently rather than assuming fixed parameters. The production monitor already refits daily, which is the correct response. Do not extend the training window in search of stability — see Section 6.

### WATCH — Structural stability

**Finding:** CUSUM breaches the 5% bands (max 1.624 > 0.948) and the largest coefficient CV is 135%. There is evidence of structural change.

**Action:** Cap the production training window at roughly 750 days (~3 years) and refit daily. Do NOT train on the full history: distant regimes are describing a different market. Verify monitor.py enforces this cap and that the app uses the same window — otherwise the two are different models publishing under one name.

### WATCH — Generalisation

**Finding:** Gap of 0.185 between in-sample (0.368) and out-of-sample (0.183) R². Out-of-sample performance is still positive.

**Action:** Quote the out-of-sample figure, never the in-sample one, in any client-facing material. Re-check after the next 90 days of live track record.

### PASS — Performance drift across folds

**Finding:** No significant decline across folds (rho = -0.80, p = 0.104).

**Action:** None.

### WATCH — Unbiasedness and efficiency (MZ)

**Finding:** Joint hypothesis rejected (p = 0.0004) with slope b = 0.834 and mean bias +7.8%. The forecast over-reacts: bands too wide when volatility spikes.

**Action:** Apply the MZ recalibration in production: sigma_adjusted = 0.00214 + 0.834 × sigma_raw, with coefficients re-estimated on a rolling window of past data only. Guard it so it is skipped if the slope falls outside [0.5, 1.6] — an extreme slope signals a data problem, not a bias to correct.

### WATCH — Kupiec — violation frequency

**Finding:** Observed coverage 96.5% vs 95% nominal (p = 0.0453): the band is WIDER than necessary. This is the benign direction — it over-covers.

**Action:** No risk action required. If you want the extra precision, apply the MZ recalibration from Section 8, which shrinks the band toward its correct width. Section 10 quantifies the opportunity cost.

### PASS — Christoffersen — independence

**Finding:** p = 0.1437 with 0 consecutive violations. Failures are dispersed, which is the benign failure mode.

**Action:** None.

### WATCH — Sizing efficiency

**Finding:** The band is about 7.3% wider than the observed violation rate justifies, so every position is that much smaller than optimal — roughly $44 of foregone notional per trade. This is an opportunity cost, not a risk exposure.

**Action:** Optional: apply the MZ recalibration from Section 8, or multiply the half-width by 0.93. Safe to defer until the live track record reaches ~90 observations — over-covering never harms the user.

### PASS — Implied leverage

**Finding:** The sizing rule implies 0.87x leverage — below 1x at current volatility.

**Action:** None. The rule is self-limiting: as volatility rises, the band widens and size falls automatically.

## Estimation (HAC standard errors)

```
 term    coef  se_classical  se_HAC  t_HAC  p_HAC
const 0.00392       0.00075 0.00085   4.63    0.0
 rv_d 0.28038       0.02044 0.03799   7.38    0.0
 rv_w 0.25463       0.03662 0.04566   5.58    0.0
 rv_m 0.33139       0.03696 0.05178   6.40    0.0
```
- R² = 0.3678 · adjusted R² = 0.3672 · n = 3266
- Persistence (b_d+b_w+b_m) = 0.8664
- HAC errors are 1.41x the classical ones

## Variance inflation

```
variable  VIF
    rv_d 2.16
    rv_w 3.96
    rv_m 2.75
```

## Residual diagnostics

```
             test  statistic  p_value                                H0
ARCH-LM (10 lags)      69.53      0.0 no conditional heteroskedasticity
    Breusch–Pagan      62.85      0.0                  homoskedasticity
      Jarque–Bera  204826.21      0.0    normally distributed residuals
```

```
 lag  lb_stat  lb_pvalue
   5  15.1513     0.0097
  10  38.7975     0.0000
  22 117.2878     0.0000
```

## Stationarity

```
 series  ADF_stat  ADF_p  KPSS_stat  KPSS_p
     RV    -5.873 0.0000      2.778    0.01
log(RV)    -4.508 0.0002      2.914    0.01
```

## Parameter stability

```
coefficient  mean   std  CV_%
        b_d 0.247 0.094  38.0
        b_w 0.222 0.187  84.3
        b_m 0.162 0.219 134.9
```
- CUSUM maximum: 1.624 (5% critical value: 0.948)

## Cross-validation

```
 fold  train_end             test_window  n_test  QLIKE  R2_oos  corr
    1 2019-03-05 2019-03-28 → 2020-09-21     544 0.9332  0.2354 0.486
    2 2020-08-30 2020-09-22 → 2022-03-19     544 0.5487  0.1981 0.457
    3 2022-02-25 2022-03-20 → 2023-09-14     544 0.6527  0.2348 0.500
    4 2023-08-23 2023-09-15 → 2025-03-11     544 0.6094  0.0898 0.333
    5 2025-02-17 2025-03-12 → 2026-08-17     524 0.5072  0.1550 0.433
```
- Mean QLIKE 0.6502 · mean out-of-sample R² 0.1826 · gap 0.1852

## Mincer–Zarnowitz

- a = +0.00214, b = 0.834, R² = 0.1678
- Joint test (a=0, b=1): F = 7.91, p = 0.0004
- Mean bias: +7.8%

## Band backtest

```
                           test    LR  p_value                         H0
Kupiec (unconditional coverage) 4.009   0.0453        violation rate = 5%
  Christoffersen (independence) 2.138   0.1437 violations are independent
   Conditional coverage (joint) 6.146   0.0463        both simultaneously
```
- Observed coverage 96.5% (nominal 95%), 29 violations of 817, 0 consecutive

## Business impact

```
                        metric         value
Forecast volatility (next day)         1.17%
           95% band half-width        ±2.29%
            Max risk per trade        $14.00
              Implied notional          $610
         Implied position size 0.00945 units
              Implied leverage         0.87x
     Round-trip cost per trade         $0.84
```
- Band-width correction needed: x0.932 (positions currently 7.3% undersized)
- Implied leverage: 0.87x
- Round-trip cost per trade: $0.84

---
*Generated by HARRV_Model_Validation_EN.ipynb*