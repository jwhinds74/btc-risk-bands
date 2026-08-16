# XGBit — Validación econométrica del modelo HAR-RV

- **Generado:** 2026-08-16 17:27 UTC
- **Fuente:** Binance · **Observaciones:** 3264
- **Período:** 2017-09-08 → 2026-08-15

## Tablero

| Prueba | Estado | Valor |
|---|---|---|
| Significancia de los componentes | **OK** | 3.0000 |
| Multicolinealidad | **OK** | 3.9500 |
| Autocorrelación de residuos | **AVISO** | 0.0000 |
| Heterocedasticidad condicional | **INFO** | 0.0000 |
| Estacionariedad de la RV | **AVISO** | 0.0000 |
| Estabilidad estructural (CUSUM) | **AVISO** | 1.6239 |
| Generalización (validación cruzada) | **AVISO** | 0.1852 |
| Insesgadez y eficiencia (Mincer–Zarnowitz) | **AVISO** | 0.0003 |
| Kupiec — frecuencia de violaciones | **FALLA** | 0.0461 |
| Christoffersen — independencia | **OK** | 0.1435 |

**Detalle:**

- *Significancia de los componentes* (OK): Los tres componentes (día, semana, mes) son significativos al 5% con errores HAC: cada horizonte aporta información propia.
- *Multicolinealidad* (OK): VIF máximo 4.0 (< 10): los componentes se solapan pero no de forma que desestabilice la estimación.
- *Autocorrelación de residuos* (AVISO): Ljung–Box a 22 rezagos: p=0.0000. Queda autocorrelación remanente. No invalida el pronóstico puntual, pero confirma que los errores estándar deben ser HAC (ya se usan) y sugiere que un componente adicional podría aportar.
- *Heterocedasticidad condicional* (INFO): ARCH-LM p=0.0000: hay heterocedasticidad condicional en los residuos. Es lo esperado al modelar volatilidad y no es un defecto: precisamente por eso las bandas se construyen con residuos estandarizados y no con un ancho fijo.
- *Estacionariedad de la RV* (AVISO): ADF rechaza raíz unitaria pero KPSS también rechaza estacionariedad: patrón típico de una serie muy persistente (memoria larga). El modelo sigue siendo utilizable, pero conviene reajustarlo con frecuencia — como ya hace el monitor diario.
- *Estabilidad estructural (CUSUM)* (AVISO): El CUSUM sale de las bandas (máx 1.624 > 0.948): hay indicios de cambio estructural. Justifica el reajuste diario del monitor y desaconseja usar ventanas de entrenamiento muy largas.
- *Generalización (validación cruzada)* (AVISO): Brecha de 0.185 entre R² dentro (0.368) y fuera (0.182) de muestra. Vigilar, aunque el desempeño fuera de muestra sigue siendo positivo.
- *Insesgadez y eficiencia (Mincer–Zarnowitz)* (AVISO): Se rechaza H0 conjunta (p=0.0003), con b=0.83. El pronóstico sobrereacciona a los cambios de volatilidad. Corregible con la recalibración de escala que ya aplica el monitor.
- *Kupiec — frecuencia de violaciones* (FALLA): Cobertura observada 96.4% vs 95% nominal (p=0.0461): se rechaza que la frecuencia sea la anunciada. La banda sobrestima el riesgo.
- *Christoffersen — independencia* (OK): p=0.1435: las violaciones no se agrupan. La banda falla de forma dispersa, que es el modo benigno.

## Estimación (errores HAC)

```
termino    coef  ee_clasico  ee_HAC  t_HAC  p_HAC
  const 0.00392     0.00075 0.00085   4.62    0.0
   rv_d 0.28036     0.02045 0.03800   7.38    0.0
   rv_w 0.25468     0.03663 0.04566   5.58    0.0
   rv_m 0.33140     0.03698 0.05179   6.40    0.0
```
- R² = 0.3676 · R² ajustado = 0.3670 · n = 3264
- Persistencia (b_d+b_w+b_m) = 0.8664
- Los errores HAC son 1.41x los clásicos

## Multicolinealidad

```
variable  VIF
    rv_d 2.15
    rv_w 3.95
    rv_m 2.74
```

## Diagnóstico de residuos

```
         test  estadistico  p_valor                                 H0
 ARCH-LM (10)        69.48      0.0 sin heterocedasticidad condicional
Breusch–Pagan        62.78      0.0                   homocedasticidad
  Jarque–Bera    204477.58      0.0             normalidad de residuos
```

```
 rezago  lb_stat  lb_pvalue
      5  15.1397     0.0098
     10  38.7562     0.0000
     22 117.1217     0.0000
```

## Estacionariedad

```
  serie  ADF_stat  ADF_p  KPSS_stat  KPSS_p
     RV    -5.869 0.0000      2.772    0.01
log(RV)    -4.483 0.0002      2.905    0.01
```

## Estabilidad de coeficientes

```
coeficiente  media  desv  CV_%
        b_d  0.247 0.094  38.0
        b_w  0.222 0.187  84.3
        b_m  0.162 0.219 134.9
```
- CUSUM máximo: 1.624 (límite 5%: 0.948)

## Validación cruzada temporal

```
 fold entrena_hasta                  prueba  n_prueba  QLIKE  R2_oos  corr
    1    2019-03-05 2019-03-28 → 2020-09-21       544 0.9332  0.2354 0.486
    2    2020-08-30 2020-09-22 → 2022-03-19       544 0.5487  0.1981 0.457
    3    2022-02-25 2022-03-20 → 2023-09-14       544 0.6527  0.2348 0.500
    4    2023-08-23 2023-09-15 → 2025-03-11       544 0.6094  0.0898 0.333
    5    2025-02-17 2025-03-12 → 2026-08-15       522 0.5076  0.1537 0.432
```
- QLIKE medio 0.6503 · R² OOS medio 0.1824

## Mincer–Zarnowitz

- a = +0.00215, b = 0.833, R² = 0.1666
- Test conjunto (a=0, b=1): F = 8.07, p = 0.0003
- Sesgo medio: +7.9%

## Backtest de bandas

```
                            test    LR  p_valor                             H0
Kupiec (cobertura incondicional) 3.978   0.0461 frecuencia de violaciones = 5%
  Christoffersen (independencia) 2.140   0.1435     violaciones independientes
  Cobertura condicional conjunta 6.119   0.0469                 ambas a la vez
```
- Cobertura observada 96.4% (nominal 95%), 29 violaciones de 816, 0 consecutivas

## Veredicto

🔴 REVISAR ANTES DE OPERAR. Fallan contrastes críticos: Kupiec — frecuencia de violaciones. Son los que gobiernan la fiabilidad de la banda como presupuesto de riesgo.

---
*Adjuntar este archivo al chat para su análisis.*