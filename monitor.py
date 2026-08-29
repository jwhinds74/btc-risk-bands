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
import time

import requests
from scipy import stats as sps
from scipy.optimize import nnls

try:
    import alerts as ALERTAS
except Exception:                      # el monitor debe correr aunque falte el modulo
    ALERTAS = None

# Motor HAR-RV: solo numpy/pandas. Sin xgboost ni scikit-learn.

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# FRECUENCIA DE BARRA (v1.7)
# ---------------------------------------------------------------------------
# Se parametriza en vez de fijarse, para poder correr 1d y 4h EN PARALELO y
# decidir con datos cual estima mejor la volatilidad — el mismo criterio que
# retiro a XGBoost. Cambiar BAR aqui reconfigura todo lo demas.
#
# El HAR de Corsi define sus tres componentes en DIAS (1, 5, 22). A barra
# intradiaria hay que traducirlos a numero de barras, no dejarlos en 1/5/22:
# a 4h, "un dia" son 6 barras y "un mes" 132. Mantenerlos sin escalar
# convertiria el componente mensual en uno de 3,7 dias.
#
# Ficheros SEPARADOS por frecuencia: mezclar evaluaciones de 1d y 4h en el mismo
# health_log destruiria el track record de ambos.
BAR = os.environ.get('BAR_INTERVAL', '1d')      # '1d' | '4h' | '8h'

_BARS_PER_DAY = {'1d': 1, '8h': 3, '4h': 6, '1h': 24}
BPD = _BARS_PER_DAY.get(BAR, 1)
_SUF = '' if BAR == '1d' else f'_{BAR}'

CONFIG = {
    'BAR': BAR,
    'BARS_PER_DAY': BPD,
    'PRICE_CACHE': f'btc_ohlcv_cache{_SUF}.csv',
    'HEALTH_LOG': f'monitoring/health_log{_SUF}.csv',
    'LAST_FORECAST': f'monitoring/last_forecast{_SUF}.json',
    'HEARTBEAT': f'monitoring/last_run{_SUF}.json',
    # Componentes HAR escalados a barras
    'HAR_W': 5 * BPD,            # componente "semanal"
    'HAR_M': 22 * BPD,           # componente "mensual"
    'CALIB_DAYS': 250 * BPD,     # cola para los residuos estandarizados
    'TRAIN_WINDOW': 750 * BPD,   # ~3 anos en barras
    'MZ_WINDOW': 500 * BPD,      # ventana para la recalibracion Mincer-Zarnowitz
    'N_BOOTSTRAP': 1000,
    'ROLLING_WINDOW': 90 * BPD,
    'CALIB_FRACTION': 0.20,   # cola held-out para los residuos conformales de la banda
    'MIN_SIGNALS_FOR_ACC_ALERT': 60,
    'COVERAGE_ALERT': 88.0,
    'KS_ALERT': 0.05,
    'REGIME_MIN_SIZE': 30 * BPD,
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
MODEL_VERSION = f'HAR-RV-4.1-{BAR}'

ALERTS = []


def log(msg):
    print(f'[monitor] {msg}', flush=True)


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def fetch_kraken(bar='1d'):
    """Respaldo publico sin API key, accesible desde EE.UU.

    Kraken solo admite ciertos intervalos (1, 5, 15, 30, 60, 240, 1440 min):
    NO existe el de 480, asi que para 8h se piden velas de 4h y se agregan de
    dos en dos. Devuelve hasta 720 velas (~120 dias a 4h).
    """
    VALIDOS = {1, 5, 15, 30, 60, 240, 1440}
    if bar == '1d':
        pedir, agregar = 1440, None
    else:
        horas = int(bar.replace('h', ''))
        minutos = horas * 60
        if minutos in VALIDOS:
            pedir, agregar = minutos, None
        else:
            divisores = [v for v in VALIDOS if minutos % v == 0 and v < minutos]
            if not divisores:
                raise RuntimeError(f'Kraken no admite ni divide el intervalo {bar}')
            pedir, agregar = max(divisores), horas

    r = requests.get('https://api.kraken.com/0/public/OHLC',
                     params={'pair': 'XBTUSD', 'interval': pedir}, timeout=25)
    r.raise_for_status()
    j = r.json()
    if j.get('error'):
        raise RuntimeError(f"Kraken (interval={pedir}): {j['error']}")
    clave = [k for k in j['result'] if k != 'last'][0]
    d = pd.DataFrame(j['result'][clave],
                     columns=['t', 'open', 'high', 'low', 'close', 'vwap', 'vol', 'n'])
    d['timestamp'] = pd.to_datetime(d['t'], unit='s')
    d = d.set_index('timestamp')[['close', 'high', 'low']].astype(float)
    d = d[~d.index.duplicated(keep='last')].sort_index()
    if agregar:
        d = d.resample(f'{agregar}h', label='left', closed='left').agg(
            {'close': 'last', 'high': 'max', 'low': 'min'}).dropna()
    if len(d) < 100:
        raise RuntimeError(f'Kraken devolvio solo {len(d)} velas de {bar}')
    return d


def cargar_cache():
    """Historia acumulada en el repo. Es la FUENTE PRINCIPAL, no un ultimo recurso.

    Cada corrida anade solo las velas nuevas y hace commit, asi que el archivo
    crece por si solo. A los pocos meses contiene mas historia intradiaria de la
    que ninguna API gratuita entrega de una vez.
    """
    if not os.path.exists(CONFIG['PRICE_CACHE']):
        return None
    try:
        d = pd.read_csv(CONFIG['PRICE_CACHE'], parse_dates=[0], index_col=0)
        for c in ('high', 'low'):
            if c not in d.columns:
                d[c] = d['close']
        d = d[['close', 'high', 'low']].astype(float)
        d = d[~d.index.duplicated(keep='last')].sort_index()
        return d[(d > 0).all(axis=1)]
    except Exception as e:
        log(f'Cache ilegible ({type(e).__name__}) — se ignora')
        return None


def descargar_recientes():
    """Solo las velas RECIENTES, no la historia completa.

    Cambio clave para viabilidad comercial: antes cada corrida bajaba ~10 paginas
    (la ventana entera), lo que agotaba en dos dias la cuota mensual de
    CryptoCompare (100 llamadas/mes en el plan gratuito; el contador marcaba 238
    de 100 consumidas). Con el cache haciendo de historia, basta una peticion
    para las ultimas velas.

    Orden: Kraken primero por ser gratuito y sin limite practico; CryptoCompare
    despues solo si queda cuota; Binance al final porque bloquea IPs de EE.UU.,
    donde corren los runners de GitHub.
    """
    errores = []

    # --- 1. Kraken: gratis, sin key, hasta 720 velas ---
    try:
        d = fetch_kraken(CONFIG['BAR'])
        return d, f"Kraken ({CONFIG['BAR']})", errores
    except Exception as e:
        errores.append(f'Kraken: {type(e).__name__}: {str(e)[:80]}')

    # --- 2. CryptoCompare: UNA sola pagina (las velas recientes) ---
    try:
        key = os.environ.get('CRYPTOCOMPARE_API_KEY', '')
        headers = {'authorization': f'Apikey {key}'} if key else {}
        if CONFIG['BAR'] == '1d':
            url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        else:
            url = 'https://min-api.cryptocompare.com/data/v2/histohour'
        r = requests.get(url, params={'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000},
                         headers=headers, timeout=25)
        r.raise_for_status()
        j = r.json()
        if j.get('Response') != 'Success':
            raise RuntimeError(j.get('Message', 'respuesta inesperada'))
        d = pd.DataFrame(j['Data']['Data'])
        d['timestamp'] = pd.to_datetime(d['time'], unit='s')
        d = d.set_index('timestamp')[['close', 'high', 'low']].astype(float)
        d = d[(d > 0).all(axis=1)].sort_index()
        if CONFIG['BAR'] != '1d':
            _h = int(CONFIG['BAR'].replace('h', ''))
            d = d.resample(f'{_h}h', label='left', closed='left').agg(
                {'close': 'last', 'high': 'max', 'low': 'min'}).dropna()
        return d, f"CryptoCompare ({CONFIG['BAR']})", errores
    except Exception as e:
        errores.append(f'CryptoCompare: {type(e).__name__}: {str(e)[:80]}')

    # --- 3. Binance: bloqueado desde EE.UU., util solo en ejecucion local ---
    try:
        r = requests.get('https://api.binance.com/api/v3/klines',
                         params={'symbol': 'BTCUSDT', 'interval': CONFIG['BAR'],
                                 'limit': 1000}, timeout=20)
        r.raise_for_status()
        k = r.json()
        d = pd.DataFrame(k, columns=['t', 'o', 'high', 'low', 'close', 'v',
                                     'ct', 'q', 'n', 'tb', 'tq', 'ig'])
        d['timestamp'] = pd.to_datetime(d['t'], unit='ms')
        d = d.set_index('timestamp')[['close', 'high', 'low']].astype(float)
        return d.sort_index(), f"Binance ({CONFIG['BAR']})", errores
    except Exception as e:
        errores.append(f'Binance: {type(e).__name__}')

    return None, None, errores


def fetch_prices():
    """Cache acumulativo + descarga incremental.

    El cache del repo aporta la historia; la API solo las velas nuevas. Ambos se
    fusionan y el resultado se vuelve a guardar, de modo que la historia crece
    con cada corrida en vez de redescargarse entera.
    """
    cache = cargar_cache()
    nuevas, source, errores = descargar_recientes()

    if nuevas is None and cache is None:
        raise RuntimeError(
            f"Sin datos para barra {CONFIG['BAR']}: fallaron todas las fuentes "
            f"({'; '.join(errores)}) y no hay cache en el repo.")

    if nuevas is None:
        log(f"Descarga fallida ({'; '.join(errores)}) — se opera con el cache")
        df, source = cache, f"cache del repo ({len(cache)} velas)"
        agregadas = 0
    elif cache is None:
        df, agregadas = nuevas, len(nuevas)
        log(f'Sin cache previo: se inicia con {len(nuevas)} velas de {source}')
    else:
        antes = len(cache)
        df = pd.concat([cache, nuevas])
        df = df[~df.index.duplicated(keep='last')].sort_index()
        agregadas = len(df) - antes
        log(f'Cache {antes} + {len(nuevas)} descargadas -> {len(df)} velas '
            f'(+{agregadas} nuevas) via {source}')

    # solo barras cerradas
    ahora = pd.Timestamp(dt.datetime.now(dt.timezone.utc)).tz_localize(None)
    if CONFIG['BAR'] == '1d':
        corte = pd.Timestamp(ahora.date())
    else:
        corte = ahora.floor(f"{int(CONFIG['BAR'].replace('h', ''))}h")
    df = df[df.index < corte]

    if len(df) < 120:
        raise RuntimeError(f'Solo {len(df)} velas de {CONFIG["BAR"]}: insuficiente')

    # Se reescribe el cache YA FUSIONADO: asi la historia se acumula.
    try:
        df.to_csv(CONFIG['PRICE_CACHE'])
    except Exception as e:
        log(f'No se pudo guardar el cache ({type(e).__name__})')

    log(f'Precios: {len(df)} velas via {source} (hasta {df.index[-1]})')
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
    d['rv_w'] = d['rv'].rolling(CONFIG['HAR_W']).mean().shift(1)
    d['rv_m'] = d['rv'].rolling(CONFIG['HAR_M']).mean().shift(1)
    d['y'] = d['rv']
    return d.dropna()


def fit_har(d_train):
    """Ajuste con coeficientes NO NEGATIVOS (v1.6).

    Con minimos cuadrados sin restriccion, la colinealidad entre los tres
    componentes producia coeficientes diarios NEGATIVOS (b_d = -0,078 en
    produccion): el modelo restaba la volatilidad de ayer justo cuando mas
    deberia pesar. Restringir a no negativos es practica estandar en HAR por
    esta razon. Si NNLS fallara, se cae a lstsq y se recorta a cero.
    """
    A = np.c_[np.ones(len(d_train)), d_train[['rv_d', 'rv_w', 'rv_m']].values]
    y = d_train['y'].values
    try:
        coef, _ = nnls(A, y)
        return coef
    except Exception:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return np.maximum(coef, 0.0)


def predict_har(coef, d_rows):
    A = np.c_[np.ones(len(d_rows)), d_rows[['rv_d', 'rv_w', 'rv_m']].values]
    return np.maximum(A @ coef, EPS)




# ---------------------------------------------------------------------------
# PISO DE VOLATILIDAD REACTIVO (v1.6)
# ---------------------------------------------------------------------------
# Problema medido: tras un salto de volatilidad, el HAR tarda semanas en
# absorberlo y la banda se queda corta. En pruebas sobre 8 semillas con shock
# inyectado, la cobertura post-shock caia al 90,1%.
#
# Diagnostico: con volatilidad perfecta ("oraculo") la cobertura llegaba al
# 100%, asi que el fallo era de ESTIMACION de volatilidad, no de deriva ni de
# colas. Un piso EWMA de 10 dias resulto demasiado lento; el de 3 dias reacciona
# a tiempo sin sobrecubrir en calma.
#
#   variante                cobertura shock   cobertura calma
#   HAR solo                     90,1%             91,5%
#   max(HAR, EWMA-10)            94,6%             92,5%
#   max(HAR, EWMA-3)             95,2%             94,0%   <- elegida
#   max(HAR, max RV 3d)          99,0%             96,7%   (sobrecubre)
#
# No predice nada: solo impide que el estimador se quede por debajo de lo que
# el mercado ACABA de mostrar.
VOL_FLOOR_SPAN = 3 * BPD    # 3 DIAS de memoria, no 3 barras


def piso_volatilidad(d, span=VOL_FLOOR_SPAN):
    """EWMA corta de la volatilidad realizada, desplazada un dia (sin mirar t)."""
    return d['rv'].ewm(span=span).mean().shift(1)


def sigma_con_piso(sigma_har, d, idx=-1, span=VOL_FLOOR_SPAN):
    """max(HAR, EWMA corta). Devuelve tambien si el piso estuvo activo."""
    try:
        piso = float(piso_volatilidad(d, span).iloc[idx])
    except Exception:
        return float(sigma_har), False
    if not np.isfinite(piso) or piso <= 0:
        return float(sigma_har), False
    return (float(max(sigma_har, piso)), bool(piso > sigma_har))


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
    sigma_har = float(predict_har(coef, fila)[0])
    sigma, piso_activo = sigma_con_piso(sigma_har, d)
    z = rng.choice(np.asarray(z_pool, dtype=float), size=n_sim)
    precios = last_close * np.exp(drift + z * sigma)
    return (float(np.percentile(precios, 2.5)), float(np.percentile(precios, 97.5)),
            float(np.median(precios)), sigma, sigma_har, piso_activo)


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

    # --- PASO 1b: ¿AVANZARON LOS DATOS? -----------------------------------
    # Si la ultima barra es la misma que en la corrida anterior, el monitor esta
    # leyendo un cache congelado: emite bandas que nunca podran evaluarse porque
    # el cierre real no llega. Producia decenas de filas con 'dentro_banda_95'
    # vacio y un track record que no crecia, sin senal de alarma alguna.
    datos_estancados = False
    if os.path.exists(CONFIG['HEALTH_LOG']):
        try:
            _prev_log = pd.read_csv(CONFIG['HEALTH_LOG'])
            if len(_prev_log):
                _ult_fecha = str(_prev_log['fecha'].iloc[-1])
                _actual = str(last_date.date() if CONFIG['BAR'] == '1d' else last_date)
                if _ult_fecha == _actual:
                    datos_estancados = True
                    log(f'DATOS ESTANCADOS: la ultima barra sigue siendo {_actual}. '
                        'La fuente no avanzo desde la corrida anterior.')
                    ALERTS.append(
                        f'Datos estancados: la ultima barra sigue siendo {_actual}. '
                        'El monitor no puede evaluar nada y el track record no crece. '
                        'Revisar las fuentes de precio.')
        except Exception:
            pass

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
        _id_actual = str(last_date.date() if CONFIG['BAR'] == '1d' else last_date)
        if target == _id_actual:
            lo, hi = float(prev['banda_lo_95']), float(prev['banda_hi_95'])
            dentro_banda = bool(lo <= last_close <= hi)
            sigma_emitida = float(prev.get('sigma', float('nan')))
            rv_real = float(d['rv'].iloc[-1])
            if np.isfinite(sigma_emitida) and sigma_emitida > 0:
                error_vol = round(float(rv_real/sigma_emitida - 1)*100, 1)
            log(f'Evaluacion de ayer -> cierre {last_close:,.0f} | banda [{lo:,.0f}, {hi:,.0f}] '
                f'-> dentro={dentro_banda} | vol emitida {sigma_emitida:.4f} vs real {rv_real:.4f}')
        else:
            log(f'last_forecast apunta a {target} pero la ultima barra es {_id_actual}: '
                'sin evaluacion en esta corrida (probable hueco en la cadena).')

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
    banda_lo, banda_hi, centro, sigma_cruda, sigma_har_raw, piso_activo = emit_band(
        coef, d_fit, z_pool, last_close, drift=0.0)
    if piso_activo:
        log(f'Piso de volatilidad ACTIVO: HAR daba {sigma_har_raw*100:.2f}%/dia, '
            f'el piso EWMA-{VOL_FLOOR_SPAN} eleva a {sigma_cruda*100:.2f}%/dia')
    sigma_hoy = max(mz_a + mz_b * sigma_cruda, EPS)
    if sigma_cruda > EPS and abs(sigma_hoy - sigma_cruda) > 1e-9:
        _esc = sigma_hoy / sigma_cruda
        banda_lo = last_close * np.exp(np.log(banda_lo / last_close) * _esc)
        banda_hi = last_close * np.exp(np.log(banda_hi / last_close) * _esc)

    log(f'HAR-RV ajustado sobre {len(d_fit)} dias: c={coef[0]:.5f} b_d={coef[1]:.3f} '
        f'b_w={coef[2]:.3f} b_m={coef[3]:.3f} | {len(z_pool)} residuos')
    log(f'Recalibracion MZ: a={mz_a:+.5f} b={mz_b:.3f} -> sigma {sigma_cruda:.5f} '
        f'-> {sigma_hoy:.5f}')

    _paso = pd.Timedelta(days=1) if CONFIG['BAR'] == '1d' \
        else pd.Timedelta(hours=int(CONFIG['BAR'].replace('h', '')))
    next_ts = last_date + _paso
    next_date = next_ts.date() if CONFIG['BAR'] == '1d' else next_ts
    json.dump({
        'emitted_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'target_date': str(next_date),
        'base_close': last_close,
        'base_date': str(last_date.date() if CONFIG['BAR'] == '1d' else last_date),
        'bar': CONFIG['BAR'],
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
        'fecha': str(last_date.date() if CONFIG['BAR'] == '1d' else last_date),
        'bar': CONFIG['BAR'],
        'cierre': round(last_close, 2),
        'dentro_banda_95': dentro_banda,
        'sigma_emitida': None if sigma_emitida is None else round(sigma_emitida, 6),
        'error_vol_%': error_vol,
        'banda_lo_95': None if prev is None else prev.get('banda_lo_95'),
        'banda_hi_95': None if prev is None else prev.get('banda_hi_95'),
        'sigma_hoy': round(sigma_hoy, 6),
        'mz_b': round(mz_b, 4),
        'train_window': int(len(d_fit)),
        'sigma_har_crudo': round(sigma_har_raw, 6),
        'piso_activo': bool(piso_activo),
        'b_d': round(float(coef[1]), 4),
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

    cols = ['fecha', 'bar', 'cierre', 'dentro_banda_95', 'sigma_emitida', 'error_vol_%',
            'cobertura_rodante_90d', 'error_vol_medio_90d', 'error_vol_abs_90d',
            'regimen_actual', 'regimen_cambio', 'ks_drift_pvalue',
            'banda_lo_95', 'banda_hi_95', 'sigma_hoy', 'banda_lo_95_hoy',
            'banda_hi_95_hoy', 'mz_b', 'train_window', 'sigma_har_crudo',
            'piso_activo', 'b_d',
            'riesgo_nivel', 'riesgo_capas', 'riesgo_shock', 'model_version', 'fuente']
    out = pd.concat([hist_log, pd.DataFrame([row])], ignore_index=True)
    out = out.reindex(columns=[c for c in cols if c in out.columns or c in row])
    # LATIDO (v4.1). Se escribe en TODA corrida, incluso si la fila se descarta
    # por datos estancados. Sin el, "el monitor no corrio" y "corrio pero la
    # fuente no avanzo" son indistinguibles desde la app, que era justo la
    # ambiguedad que impedia diagnosticar.
    try:
        json.dump({
            'ran_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'bar': CONFIG['BAR'],
            'last_bar': str(last_date),
            'source': source,
            'stale': bool(datos_estancados),
            'n_velas': int(len(px)),
            'model_version': MODEL_VERSION,
        }, open(CONFIG['HEARTBEAT'], 'w'), indent=2)
    except Exception as e:
        log(f'No se pudo escribir el latido ({type(e).__name__})')

    if datos_estancados and len(out) > 1:
        # Se descarta la fila duplicada: repetir la misma barra inflaria el
        # conteo de "dias monitoreados" sin anadir informacion.
        out = out.iloc[:-1]
        log('Fila descartada por datos estancados (no se duplica la barra).')

    out.to_csv(CONFIG['HEALTH_LOG'], index=False)
    log(f"{CONFIG['HEALTH_LOG']} actualizado ({len(out)} filas).")

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
    _ahora = dt.datetime.now(dt.timezone.utc)
    if _ahora.weekday() == 6 and (CONFIG['BAR'] == '1d' or _ahora.hour < 4):
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
