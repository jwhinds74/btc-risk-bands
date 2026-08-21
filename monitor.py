#!/usr/bin/env python3
"""
XGBit — Monitor diario del modelo (GitHub Actions, 01:00 UTC tras el colector).

Orden de operaciones (spec V5, seccion 6):
  1. Carga caches -> construye features CORREGIDAS -> carga last_forecast.json de AYER.
  2. Compara: el cierre real cayo dentro de las bandas? la P>55% acerto?
  3. Reentrena y emite el forecast de HOY -> sobreescribe last_forecast.json.
  4. Agrega una fila a monitoring/health_log.csv.
  5. Evalua umbrales -> Telegram SOLO si cruza (+ resumen dominical siempre).
  6. El commit+push lo hace el workflow.

Todas las features llevan shift(1) en las rolling: el bug de leakage de V4 no
puede reaparecer aqui, porque este archivo es la unica fuente de verdad del
track record publicado.
"""
import os
import sys
import json
import datetime as dt

import numpy as np
import pandas as pd
import requests
from scipy import stats as sps

try:
    import alerts as ALERTAS
except Exception:                      # el monitor debe correr aunque falte el modulo
    ALERTAS = None

# Motor HAR-RV: solo numpy/pandas. Sin xgboost ni scikit-learn.

# ---------------------------------------------------------------------------
CONFIG = {
    'PRICE_CACHE': 'btc_ohlcv_cache.csv',
    'HEALTH_LOG': 'monitoring/health_log.csv',
    'LAST_FORECAST': 'monitoring/last_forecast.json',
    'CALIB_DAYS': 250,           # cola para estimar los residuos estandarizados
    'TRAIN_WINDOW': 750,         # ~3 anos: mas historia mezcla regimenes incompatibles
    'MZ_WINDOW': 500,            # ventana para la recalibracion Mincer-Zarnowitz
    'N_BOOTSTRAP': 1000,
    'ROLLING_WINDOW': 90,
    'CALIB_FRACTION': 0.20,   # cola held-out para los residuos conformales de la banda
    'MIN_SIGNALS_FOR_ACC_ALERT': 60,
    'COVERAGE_ALERT': 88.0,
    'KS_ALERT': 0.05,
    'REGIME_MIN_SIZE': 30,
    'REGIME_PEN': 10,
    'SEED': 42,
}

VOL_COL = 'vol_7'
VOL_FLOOR_Q = 0.10


def conditional_sigma(v):
    """Volatilidad condicional saneada (sin ceros ni valores absurdos)."""
    a = np.asarray(v, dtype=float)
    fin = a[np.isfinite(a) & (a > 0)]
    if len(fin) == 0:
        return np.ones_like(a)
    floor = max(np.quantile(fin, VOL_FLOOR_Q), 1e-6)
    a = np.where(np.isfinite(a), a, np.median(fin))
    return np.maximum(a, floor)


# Version del motor. CAMBIALA con cada modificacion del modelo: sin esto el
# track record mezcla versiones y la cobertura acumulada deja de significar nada.
MODEL_VERSION = 'HAR-RV-1.3'

ALERTS = []


def log(msg):
    print(f'[monitor] {msg}', flush=True)


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def fetch_prices():
    """Jerarquia identica a la app: CryptoCompare -> Binance klines -> cache."""
    df, source = None, None
    key = os.environ.get('CRYPTOCOMPARE_API_KEY', '')
    try:
        p = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000}
        headers = {'authorization': f'Apikey {key}'} if key else {}
        r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday',
                         params=p, headers=headers, timeout=20)
        r.raise_for_status()
        j = r.json()
        if j.get('Response') != 'Success':
            raise RuntimeError(j.get('Message', 'respuesta inesperada'))
        d = pd.DataFrame(j['Data']['Data'])
        d['timestamp'] = pd.to_datetime(d['time'], unit='s')
        df = d.set_index('timestamp')[['close', 'high', 'low']].astype(float)
        source = 'CryptoCompare'
    except Exception as e:
        log(f'CryptoCompare no disponible ({type(e).__name__}) -> Binance')
        try:
            rows, end = [], None
            for _ in range(3):
                p = {'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 500}
                if end:
                    p['endTime'] = end
                r = requests.get('https://api.binance.com/api/v3/klines', params=p, timeout=15)
                r.raise_for_status()
                k = r.json()
                if not k:
                    break
                rows = k + rows
                end = k[0][0] - 1
                if len(k) < 500:
                    break
            d = pd.DataFrame(rows, columns=['t', 'open', 'high', 'low', 'close', 'vol',
                                            'ct', 'volume', 'n', 'tb', 'tq', 'ig'])
            d['timestamp'] = pd.to_datetime(d['t'], unit='ms')
            df = d.set_index('timestamp')[['close', 'high', 'low']].astype(float)
            df = df[~df.index.duplicated(keep='last')].sort_index()
            source = 'Binance (klines)'
        except Exception as e2:
            log(f'Binance no disponible ({type(e2).__name__}) -> cache')
            if os.path.exists(CONFIG['PRICE_CACHE']):
                df = pd.read_csv(CONFIG['PRICE_CACHE'], parse_dates=[0], index_col=0)
                for c in ('high', 'low'):
                    if c not in df.columns:
                        df[c] = df['close']
                df = df[['close', 'high', 'low']].astype(float)
                source = 'cache local'
                ALERTS.append('Fetch de precio fallido en ambas APIs: se uso el cache local.')

    if df is None or len(df) == 0:
        raise RuntimeError('Sin datos de precio: APIs y cache fallaron.')

    df = df.sort_index()
    today_utc = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    df = df[df.index < today_utc]          # solo velas cerradas
    if source != 'cache local':
        try:
            df.to_csv(CONFIG['PRICE_CACHE'])
        except Exception:
            pass
    log(f'Precios: {len(df)} velas via {source} (hasta {df.index[-1].date()})')
    return df, source


# ---------------------------------------------------------------------------
# 2. MOTOR HAR-RV
# ---------------------------------------------------------------------------
EPS = 1e-8


def parkinson_rv(px):
    """Volatilidad diaria por rango alto-bajo (Parkinson 1980)."""
    hl = np.log(px['high'] / px['low'])
    return np.sqrt(hl ** 2 / (4 * np.log(2)))


def build_har_dataset(px):
    """Componentes diario/semanal/mensual. Todo con shift(1): la fila de t solo
    contiene informacion hasta t-1."""
    d = pd.DataFrame(index=px.index)
    d['ret'] = np.log(px['close']).diff()
    d['rv'] = parkinson_rv(px)
    d['rv_d'] = d['rv'].shift(1)
    d['rv_w'] = d['rv'].rolling(5).mean().shift(1)
    d['rv_m'] = d['rv'].rolling(22).mean().shift(1)
    d['y'] = d['rv']
    return d.dropna()


def fit_har(d_train):
    A = np.c_[np.ones(len(d_train)), d_train[['rv_d', 'rv_w', 'rv_m']].values]
    coef, *_ = np.linalg.lstsq(A, d_train['y'].values, rcond=None)
    return coef


def predict_har(coef, d_rows):
    A = np.c_[np.ones(len(d_rows)), d_rows[['rv_d', 'rv_w', 'rv_m']].values]
    return np.maximum(A @ coef, EPS)


def emit_band(coef, d, z_pool, last_close, drift=0.0, n_sim=None, seed=None):
    """Banda 95% a 1 dia: sigma del HAR x residuos estandarizados remuestreados.

    Los z conservan las colas gruesas reales en vez de asumir normalidad, y la
    sigma del dia hace que la banda respire con el regimen de volatilidad.
    """
    n_sim = n_sim or CONFIG['N_BOOTSTRAP']
    rng = np.random.default_rng(seed or CONFIG['SEED'])
    fila = pd.DataFrame([{
        'rv_d': float(d['rv'].iloc[-1]),
        'rv_w': float(d['rv'].iloc[-5:].mean()),
        'rv_m': float(d['rv'].iloc[-22:].mean()),
    }])
    sigma = float(predict_har(coef, fila)[0])
    z = rng.choice(np.asarray(z_pool, dtype=float), size=n_sim)
    precios = last_close * np.exp(drift + z * sigma)
    return (float(np.percentile(precios, 2.5)), float(np.percentile(precios, 97.5)),
            float(np.median(precios)), sigma)


def ks_drift(d):
    """KS entre la volatilidad reciente (60d) y la historica."""
    try:
        s = d['rv_d'].dropna()
        if len(s) < 200:
            return float('nan')
        return float(sps.ks_2samp(s.iloc[-60:], s.iloc[:-60]).pvalue)
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# 4. REGIMEN Y DRIFT
# ---------------------------------------------------------------------------
def detect_regime(px):
    """Devuelve (etiqueta_regimen, cambio_reciente_bool)."""
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model='rbf', min_size=CONFIG['REGIME_MIN_SIZE']).fit(px['close'].values)
        bps = algo.predict(pen=CONFIG['REGIME_PEN'])
        start = bps[-2] if len(bps) >= 2 else 0
        seg = px['close'].iloc[start:]
        vol = seg.pct_change().std() * 100
        all_vols = []
        prev = 0
        for bp in bps:
            s = px['close'].iloc[prev:bp]
            if len(s) > 2:
                all_vols.append(s.pct_change().std() * 100)
            prev = bp
        med = np.median(all_vols) if all_vols else vol
        label = 'estable' if vol <= med else 'volatil'
        days_in_regime = len(seg)
        # "cambio" = el quiebre mas reciente ocurrio en los ultimos 3 dias
        changed = days_in_regime <= 3
        return f'{label}({days_in_regime}d)', bool(changed)
    except Exception as e:
        log(f'Deteccion de regimen no disponible: {type(e).__name__}')
        return 'n/d', False


# ---------------------------------------------------------------------------
# 5. TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not token or not chat:
        log('Telegram no configurado (faltan secrets) — mensaje solo en el log:')
        log(text)
        return
    try:
        r = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                          data={'chat_id': chat, 'text': text,
                                'parse_mode': 'Markdown'}, timeout=15)
        r.raise_for_status()
        log('Telegram enviado.')
    except Exception as e:
        log(f'Telegram fallo: {type(e).__name__}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs('monitoring', exist_ok=True)
    px, source = fetch_prices()
    d = build_har_dataset(px)
    last_date = px.index[-1]
    last_close = float(px['close'].iloc[-1])

    # --- PASO 2: evaluar el forecast de AYER contra la realidad -------------
    dentro_banda, sigma_emitida, error_vol = None, None, None
    prev = None
    if os.path.exists(CONFIG['LAST_FORECAST']):
        try:
            prev = json.load(open(CONFIG['LAST_FORECAST']))
        except Exception as e:
            log(f'last_forecast.json ilegible ({type(e).__name__}) — se omite la evaluacion.')

    if prev:
        target = prev.get('target_date')
        if target == str(last_date.date()):
            lo, hi = float(prev['banda_lo_95']), float(prev['banda_hi_95'])
            dentro_banda = bool(lo <= last_close <= hi)
            sigma_emitida = float(prev.get('sigma', float('nan')))
            rv_real = float(d['rv'].iloc[-1])
            if np.isfinite(sigma_emitida) and sigma_emitida > 0:
                error_vol = round(float(rv_real/sigma_emitida - 1)*100, 1)
            log(f'Evaluacion de ayer -> cierre {last_close:,.0f} | banda [{lo:,.0f}, {hi:,.0f}] '
                f'-> dentro={dentro_banda} | vol emitida {sigma_emitida:.4f} vs real {rv_real:.4f}')
        else:
            log(f'last_forecast apunta a {target} pero el ultimo cierre es {last_date.date()}: '
                'sin evaluacion hoy (probable dia perdido en la cadena).')

    # --- PASO 3: reajustar HAR y emitir el forecast de HOY ------------------
    # VENTANA ACOTADA (v1.3). Antes se ajustaba sobre TODO el historial (9 anos).
    # La validacion econometrica dio CUSUM fuera de banda (1,62 vs 0,95) y un
    # coeficiente mensual con CV del 135% entre ventanas rodantes: mezclar 2017
    # con 2026 es mezclar mercados incompatibles.
    d_fit = d.tail(CONFIG['TRAIN_WINDOW']) if len(d) > CONFIG['TRAIN_WINDOW'] else d
    coef = fit_har(d_fit)
    sig_hist = predict_har(coef, d_fit)

    # RECALIBRACION MINCER-ZARNOWITZ (v1.3). La validacion dio b=0,833 y sesgo
    # +7,9%: el modelo SOBRERREACCIONA a los cambios de volatilidad. Se corrige
    # con la propia regresion MZ estimada sobre el pasado observado.
    mz_a, mz_b = 0.0, 1.0
    try:
        _w = min(CONFIG['MZ_WINDOW'], len(d_fit))
        _f = np.asarray(sig_hist[-_w:], dtype=float)
        _r = np.asarray(d_fit['y'].values[-_w:], dtype=float)
        if len(_f) >= 100 and np.std(_f) > 0:
            _A = np.c_[np.ones(len(_f)), _f]
            _c, *_ = np.linalg.lstsq(_A, _r, rcond=None)
            # Solo se aplica si la correccion es moderada; una pendiente absurda
            # indicaria un problema de datos, no un sesgo a corregir.
            if 0.5 <= _c[1] <= 1.6:
                mz_a, mz_b = float(_c[0]), float(_c[1])
    except Exception as e:
        log(f'Recalibracion MZ omitida ({type(e).__name__})')

    z_pool = (d_fit['ret'].values / np.maximum(sig_hist, EPS))
    z_pool = z_pool[np.isfinite(z_pool)][-CONFIG['CALIB_DAYS']:]

    # SIN DERIVA (v1.3): la banda se centra en el ultimo cierre. El termino de
    # deriva la desplazaba y sugeria direccion sin haber sido validado.
    banda_lo, banda_hi, centro, sigma_cruda = emit_band(coef, d_fit, z_pool,
                                                        last_close, drift=0.0)
    sigma_hoy = max(mz_a + mz_b * sigma_cruda, EPS)
    if sigma_cruda > EPS and abs(sigma_hoy - sigma_cruda) > 1e-9:
        _esc = sigma_hoy / sigma_cruda
        banda_lo = last_close * np.exp(np.log(banda_lo / last_close) * _esc)
        banda_hi = last_close * np.exp(np.log(banda_hi / last_close) * _esc)

    log(f'HAR-RV ajustado sobre {len(d_fit)} dias: c={coef[0]:.5f} b_d={coef[1]:.3f} '
        f'b_w={coef[2]:.3f} b_m={coef[3]:.3f} | {len(z_pool)} residuos')
    log(f'Recalibracion MZ: a={mz_a:+.5f} b={mz_b:.3f} -> sigma {sigma_cruda:.5f} '
        f'-> {sigma_hoy:.5f}')

    next_date = (last_date + pd.Timedelta(days=1)).date()
    json.dump({
        'emitted_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'target_date': str(next_date),
        'base_close': last_close,
        'base_date': str(last_date.date()),
        'motor': 'HAR-RV',
        'sigma': round(sigma_hoy, 6),
        'sigma_cruda': round(sigma_cruda, 6),
        'mz_a': round(mz_a, 6), 'mz_b': round(mz_b, 4),
        'train_window': int(len(d_fit)),
        'coef': [round(float(c), 6) for c in coef],
        'banda_lo_95': round(banda_lo, 2),
        'banda_hi_95': round(banda_hi, 2),
        'centro': round(centro, 2),
        'fuente': source,
    }, open(CONFIG['LAST_FORECAST'], 'w'), indent=2)
    log(f'Forecast emitido para {next_date}: vol {sigma_hoy*100:.2f}%/dia | '
        f'banda [{banda_lo:,.0f}, {banda_hi:,.0f}]')

    # --- PASO 3b: CAPAS DE ALERTA -----------------------------------------
    # Tres detectores independientes de RIESGO DE VOLATILIDAD (nunca de direccion):
    # shock ex post sobre datos propios, calendario de eventos programados, y
    # busqueda de noticias (esta ultima explicitamente NO validada).
    riesgo = None
    if ALERTAS is not None:
        try:
            rv_real_hoy = float(d['rv'].iloc[-1])
            riesgo = ALERTAS.evaluar_riesgo(
                rv_realizada=rv_real_hoy,
                sigma_pronosticada=sigma_emitida if sigma_emitida else sigma_hoy,
                usar_noticias=os.environ.get('ALERTS_NEWS', '1') == '1',
            )
            log(f"Riesgo: nivel={riesgo['nivel']} | capas activas={riesgo['n_capas']}"
                f"{' | REFORZADO' if riesgo['reforzado'] else ''}")
        except Exception as e:
            log(f'Capa de alertas omitida ({type(e).__name__}: {e})')
            riesgo = None

    # --- PASO 4: fila del health_log ---------------------------------------
    regimen, regimen_cambio = detect_regime(px)
    ks_p = ks_drift(d)

    hist_log = pd.DataFrame()
    if os.path.exists(CONFIG['HEALTH_LOG']):
        try:
            hist_log = pd.read_csv(CONFIG['HEALTH_LOG'])
        except Exception:
            hist_log = pd.DataFrame()

    row = {
        'fecha': str(last_date.date()),
        'cierre': round(last_close, 2),
        'dentro_banda_95': dentro_banda,
        'sigma_emitida': None if sigma_emitida is None else round(sigma_emitida, 6),
        'error_vol_%': error_vol,
        'banda_lo_95': None if prev is None else prev.get('banda_lo_95'),
        'banda_hi_95': None if prev is None else prev.get('banda_hi_95'),
        'sigma_hoy': round(sigma_hoy, 6),
        'mz_b': round(mz_b, 4),
        'train_window': int(len(d_fit)),
        'banda_lo_95_hoy': round(banda_lo, 2),
        'banda_hi_95_hoy': round(banda_hi, 2),
        'regimen_actual': regimen,
        'regimen_cambio': regimen_cambio,
        'ks_drift_pvalue': None if np.isnan(ks_p) else round(ks_p, 4),
        'riesgo_nivel': riesgo['nivel'] if riesgo else None,
        'riesgo_capas': riesgo['n_capas'] if riesgo else None,
        'riesgo_shock': riesgo['shock']['ratio'] if riesgo else None,
        'model_version': MODEL_VERSION,
        'fuente': source,
    }

    # Metricas rodantes 90d (incluyendo la fila de hoy)
    W = CONFIG['ROLLING_WINDOW']
    comb = pd.concat([hist_log, pd.DataFrame([row])], ignore_index=True)
    tail = comb.tail(W)

    cov = tail['dentro_banda_95'].dropna() if 'dentro_banda_95' in tail else pd.Series(dtype=float)
    row['cobertura_rodante_90d'] = round(cov.astype(bool).mean() * 100, 1) if len(cov) else None

    ev = pd.to_numeric(tail.get('error_vol_%', pd.Series(dtype=float)), errors='coerce').dropna()
    row['error_vol_medio_90d'] = round(float(ev.mean()), 1) if len(ev) else None
    row['error_vol_abs_90d'] = round(float(ev.abs().mean()), 1) if len(ev) else None

    cols = ['fecha', 'cierre', 'dentro_banda_95', 'sigma_emitida', 'error_vol_%',
            'cobertura_rodante_90d', 'error_vol_medio_90d', 'error_vol_abs_90d',
            'regimen_actual', 'regimen_cambio', 'ks_drift_pvalue',
            'banda_lo_95', 'banda_hi_95', 'sigma_hoy', 'banda_lo_95_hoy',
            'banda_hi_95_hoy', 'mz_b', 'train_window',
            'riesgo_nivel', 'riesgo_capas', 'riesgo_shock', 'model_version', 'fuente']
    out = pd.concat([hist_log, pd.DataFrame([row])], ignore_index=True)
    out = out.reindex(columns=[c for c in cols if c in out.columns or c in row])
    out.to_csv(CONFIG['HEALTH_LOG'], index=False)
    log(f'health_log.csv actualizado ({len(out)} filas).')

    # --- PASO 5: umbrales -> Telegram --------------------------------------
    if row['cobertura_rodante_90d'] is not None and row['cobertura_rodante_90d'] < CONFIG['COVERAGE_ALERT']:
        ALERTS.append(f"Cobertura 90d en {row['cobertura_rodante_90d']}% (umbral {CONFIG['COVERAGE_ALERT']}%)")
    if riesgo and riesgo['nivel'] in ('medio', 'alto'):
        _pref = 'RIESGO ELEVADO' if riesgo['nivel'] == 'alto' else 'Riesgo moderado'
        ALERTS.append(f"{_pref} de volatilidad ({riesgo['n_capas']} capa(s) activa(s))"
                      + (' — REFORZADO por coincidencia entre capas' if riesgo['reforzado'] else ''))

    if regimen_cambio:
        ALERTS.append(f'Cambio de regimen detectado ({regimen}) — revisar posiciones, no abrir tamano nuevo')
    if row['ks_drift_pvalue'] is not None and row['ks_drift_pvalue'] < CONFIG['KS_ALERT']:
        ALERTS.append(f"Drift de features: KS p={row['ks_drift_pvalue']} (<{CONFIG['KS_ALERT']})")

    if ALERTS:
        _detalle = f"\n\n{riesgo['mensaje']}" if (riesgo and riesgo['nivel'] != 'ninguno') else ''
        msg = ('🚨 *BTC Risk Bands — alerta del monitor*\n'
               f"Fecha: {row['fecha']}\n\n" + '\n'.join(f'• {a}' for a in ALERTS) + _detalle +
               f"\n\nCierre: ${last_close:,.0f} | vol manana: {sigma_hoy*100:.2f}% | "
               f"banda [{banda_lo:,.0f}, {banda_hi:,.0f}]")
        send_telegram(msg)

    # Resumen dominical (prueba de vida) — domingo UTC
    if dt.datetime.now(dt.timezone.utc).weekday() == 6:
        send_telegram(
            '📊 *BTC Risk Bands — resumen semanal*\n'
            f"Dias monitoreados: {len(out)}\n"
            f"Cobertura 90d: {row['cobertura_rodante_90d']}%\n"
            f"Error medio de vol 90d: {row['error_vol_medio_90d']}%\n"
            f"Regimen: {regimen}\n"
            f"Cierre: ${last_close:,.0f} | vol manana: {sigma_hoy*100:.2f}%\n"
            f"{'Sin alertas esta corrida.' if not ALERTS else 'CON alertas — ver arriba.'}"
        )

    log('Monitor completado.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        send_telegram(f'🚨 *BTC Risk Bands — monitor FALLO*\n`{type(e).__name__}: {e}`\n'
                      'El sistema esta ciego hasta la proxima corrida. Revisar pestana Actions.')
        sys.exit(1)
