# BTC Risk Bands — motor HAR-RV

![monitor](../../actions/workflows/monitor.yml/badge.svg)
![collector](../../actions/workflows/collect.yml/badge.svg)

Bandas de riesgo para Bitcoin con cobertura medida y track record auditable.
**No pronostica dirección.**

## Por qué existe este repo

Es la versión sin machine learning de `bitcoin-regime-forecasting`. Doce
experimentos con protocolo anti-sesgo (holdout bloqueado, test anti-leakage con
control positivo, walk-forward con costos, baselines obligatorios) concluyeron:

| Pregunta | Resultado |
|---|---|
| ¿Predice dirección? | **No.** Theil U2 = 1,046, IC95 [1,025–1,080] — peor que "mañana = hoy", con Diebold-Mariano significativo |
| ¿Sobreajusta? | **Sí.** R² in-sample +0,42 vs OOS −0,08 (brecha 0,50) |
| ¿La probabilidad discrimina? | **No.** Resolución 0,0002, AUC 0,503 |
| ¿XGBoost gana en volatilidad a 1 día? | **No.** HAR-RV mejor en QLIKE (0,4978 vs 0,5206) |
| ¿Y a 8h con más potencia? | **No concluyente.** XGBoost +2,7% pero p=0,31 con n=3.000; al triplicar la potencia el p-valor empeoró (0,244 → 0,311) |

Conclusión: XGBoost no aportaba en ningún frente. Esta versión lo retira.

## El motor

```
RV_{t+1} = c + b_d·RV_t + b_w·RV̄_5 + b_m·RV̄_22
```

HAR-RV (Corsi, 2009). Cuatro parámetros por mínimos cuadrados. Captura el hecho
estilizado central de los mercados: la volatilidad se agrupa y tiene memoria
larga, con operadores de horizontes distintos superpuestos.

La volatilidad realizada se estima con el **rango de Parkinson** (alto/bajo), que
sobre BTC real tiene ruido relativo ~0,53 frente a ~0,89 del retorno absoluto.

Las bandas se construyen simulando 1.000 caminos: cada retorno es
`drift + z·σ_HAR(t)`, donde los `z` son residuos estandarizados remuestreados de
la historia — conservan las colas gruesas reales en vez de asumir normalidad.

## Frente a la versión XGBoost

| | XGBoost | HAR-RV |
|---|---|---|
| Líneas de `app.py` | ~3.400 | ~1.300 |
| Ajustes por clic | 78 | 1 (`lstsq`) |
| Dependencias pesadas | xgboost, sklearn, arch | ninguna |
| Hiperparámetros | GridSearch + RandomizedSearch | ninguno |
| Sobreajuste posible | sí (brecha medida 0,50) | no, con 4 parámetros |
| Throttling de CPU | sí | no |

## Estructura

```
├── app.py                      # Streamlit Cloud despliega desde aquí
├── collect_data.py             # colector diario
├── monitor.py                  # chequeo diario del modelo
├── monitoring/
│   ├── health_log.csv          # 1 fila/día (lo escribe monitor.py)
│   └── last_forecast.json      # banda y sigma emitidas hoy
└── .github/workflows/
    ├── collect.yml             # 00:40 UTC
    └── monitor.yml             # 01:00 UTC
```

## Setup

1. Subir el repo y apuntar Streamlit Cloud a `app.py`.
2. **Secrets de GitHub** (Settings → Secrets → Actions): `TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_CHAT_ID`, y opcionalmente `CRYPTOCOMPARE_API_KEY`.
3. **Secrets de Streamlit Cloud**: el bloque `[credentials]` de autenticación.
4. Copiar `btc_ohlcv_cache.csv` desde el repo XGBoost — le da historia al monitor
   desde el día uno y hace que ambas apps evalúen sobre los mismos días.
5. **No** copiar `monitoring/health_log.csv`: cada motor arranca su track record
   limpio. Mezclarlos invalidaría la cobertura acumulada de ambos.
6. Actions → *Run workflow* manual, primero el colector y después el monitor.

## Qué se registra cada día

`health_log.csv`: cierre, si cayó dentro de la banda 95%, la sigma emitida el día
anterior frente a la volatilidad realmente realizada, el error porcentual de esa
estimación, cobertura rodante a 90 días, régimen vigente y p-valor del test KS de
drift.

Alertas por Telegram solo si algo cruza umbral (cobertura 90d < 88%, cambio de
régimen, drift, fetch fallido), más un resumen dominical como prueba de vida.

## Comparación en vivo

Este repo corre en paralelo con el de XGBoost. Ambos emiten bandas para los mismos
días, con track records sellados por separado. A los ~90 días la comparación de
cobertura decide cuál sobrevive — con evidencia en vivo, no solo con backtest.

---
© 2026 H&A DataLab Solutions SPA · Hinds-Analytics.com
