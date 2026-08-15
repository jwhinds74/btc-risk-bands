"""
BTC Risk Bands — motor HAR-RV
=============================
Version derivada de XGBit V5.3 tras 12 experimentos de validacion.

QUE CAMBIO Y POR QUE
--------------------
XGBoost fue retirado. Resumen de la evidencia:
  * Direccion: Theil U2 = 1,046 (IC95 [1,025-1,080]) -> PEOR que "manana = hoy",
    con Diebold-Mariano significativo. No hay senal direccional.
  * Sobreajuste: R2 in-sample +0,42 vs out-of-sample -0,08 (brecha 0,50).
  * Volatilidad a 1 dia: HAR-RV gano en QLIKE (0,4978 vs 0,5206).
  * Volatilidad a 8h con n_test=3.000: XGBoost lidero +2,7% pero p=0,31.
    Al triplicar la potencia el p-valor EMPEORO (0,244 -> 0,311): firma de
    fluctuacion muestral, no de efecto real.

Lo que se conserva de la version anterior: bandas conformales normalizadas por
volatilidad (cobertura medida 91,1% a 95% nominal y 68,9% a 68% nominal),
deteccion de regimenes, contexto tecnico descriptivo y track record auditable.

Lo que se elimina: XGBoost, GridSearchCV, RandomizedSearchCV, el clasificador de
probabilidad direccional y su calibracion isotonica (resolucion medida 0,0002,
AUC 0,503: la P no discriminaba), y la cache de modelos.

De ~3.400 lineas a ~900. De 78 ajustes de modelo por clic a un lstsq.
"""
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# ============================================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="BTC Risk Bands | Bitcoin Regime Forecasting",
    page_icon="assets/XGBit_Logo.JPG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

# Load credentials from secrets (Streamlit Cloud) or local config
try:
    # For Streamlit Cloud - uses st.secrets
    config = {
        'credentials': {
            'usernames': dict(st.secrets.get("credentials", {}).get("usernames", {}))
        },
        'cookie': {
            'name': st.secrets.get("cookie", {}).get("name", "xgbit_auth"),
            'key': st.secrets.get("cookie", {}).get("key", "xgbit_secret_key_2024"),
            'expiry_days': st.secrets.get("cookie", {}).get("expiry_days", 30)
        }
    }
except Exception:
    # Fallback for local development
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'name': 'Administrador',
                    'email': 'analytics.hinds@gmail.com',
                    'password': '$2b$12$LQv3c1yqBwEHh4kJYqEZeOQZQZQZQZQZQZQZQZQZQZQZQZQZQZQZQ'  # Placeholder
                }
            }
        },
        'cookie': {
            'name': 'xgbit_auth',
            'key': 'xgbit_secret_key_2024',
            'expiry_days': 30
        }
    }

# Create authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ============================================================================
# LOGIN PAGE STYLING
# ============================================================================
login_css = """
<style>
    /* Login page background */
    [data-testid="stAppViewContainer"] {
        background-color: #525252;
    }
    
    /* Hide sidebar on login */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Center login container */
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 40px;
        background-color: #5A5A5A;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .login-title {
        color: #F5A05A;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .login-subtitle {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .login-description {
        color: #E0E0E0;
        font-size: 14px;
        margin-bottom: 25px;
    }
    
    .login-features {
        color: #B0B0B0;
        font-size: 13px;
        text-align: left;
        margin: 20px 0;
        padding: 15px;
        background-color: #4A4A4A;
        border-radius: 8px;
    }
    
    .login-contact {
        color: #909090;
        font-size: 12px;
        margin-top: 20px;
    }
    
    .login-contact a {
        color: #F5A05A;
    }
    
    .powered-by {
        color: #707070;
        font-size: 11px;
        margin-top: 30px;
    }
</style>
"""

# ============================================================================
# AUTHENTICATION FLOW
# ============================================================================

# First, show the welcome content (logo, info) for non-authenticated users
# We need to check auth status first without rendering the form

# Initialize session state for authentication if not exists
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Get current auth status
authentication_status = st.session_state.get('authentication_status', None)

if authentication_status != True:
    # Clear forecast data when not authenticated
    keys_to_clear = ['forecast_generated', 'forecast_data', 'forecast_df', 'forecasts', 
                     'regime_perf_data', 'walk_forward_computed', 'num_regimes',
                     'metrics', 'hist', 'hist_regime', 'breakpoints', 'regime_stats',
                     'regime_info', 'messages']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Not authenticated - show welcome page with login
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Logo centered at TOP
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            import base64
            with open("assets/XGBit_Logo.JPG", "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            st.markdown(f"<div style='text-align: center;'><img src='data:image/jpeg;base64,{logo_data}' width='200' style='border-radius: 15px;'></div>", unsafe_allow_html=True)
        except: 
            pass
    
    st.markdown("<h1 style='text-align: center; color: #F5A05A; font-size: 48px; margin-bottom: 5px;'>XGBit</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #FFFFFF; font-size: 22px; margin-bottom: 20px;'>Bitcoin Regime-Aware Forecasting Platform</p>", unsafe_allow_html=True)
    
    # Features box
    st.markdown("""
    <div style='max-width: 450px; margin: 0 auto; padding: 25px; background-color: #4A4A4A; border-radius: 10px; text-align: left;'>
        <p style='color: #E0E0E0; font-size: 18px; margin: 10px 0;'>📐 Bandas de riesgo con cobertura empírica medida</p>
        <p style='color: #E0E0E0; font-size: 18px; margin: 10px 0;'>📊 Motor HAR-RV — cuatro parámetros, sin machine learning</p>
        <p style='color: #E0E0E0; font-size: 18px; margin: 10px 0;'>🔍 Detección de regímenes y contexto técnico descriptivo</p>
        <p style='color: #E0E0E0; font-size: 18px; margin: 10px 0;'>🩺 Track record auditable, sellado con commits diarios</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #F5A05A; font-size: 16px; margin-top: 30px;'>🔐 Private Access</p>", unsafe_allow_html=True)
    
    # Login form at BOTTOM (centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name, authentication_status, username = authenticator.login('main')
    
    # Show error if login failed
    if authentication_status == False:
        st.error('❌ Incorrect username or password')
    
    # Contact info
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <p style='color: #909090; font-size: 18px;'>
            Don't have an account? Contact: <a href='https://docs.google.com/forms/d/e/1FAIpQLSdXaPtX28px7kXow-JcQhTOqqzuhkgovLhyIvTORzXoYMTE_g/viewform' target='_blank' style='color: #F5A05A;'>Solicitar acceso</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset Session button (for cookie/session issues)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Reset Session", help="Use if login form doesn't appear or you experience issues", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Clear cookies by resetting auth status
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.rerun()
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 30px;'>
        <p style='color: #909090; font-size: 18px;'>Powered by Hinds Analytics</p>
        <p style='color: #808080; font-size: 18px; margin-top: 20px;'>
            © 2026 H&A DataLab Solutions SPA (RUT: 77.905.218-4).<br>
            All rights reserved. Developed by Hinds-Analytics.com.<br>
            Copying or use without permission is prohibited.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# User is authenticated - get name
name = st.session_state.get('name', 'User')
username = st.session_state.get('username', '')

# ============================================================================
# USER IS AUTHENTICATED - SHOW MAIN APP
# ============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

# Sin xgboost, sin scikit-learn, sin arch: el motor HAR-RV solo necesita
# numpy/pandas. Menos dependencias = menos superficie de fallo en el deploy
# y arranque mucho mas rapido en Streamlit Cloud.

# ============================================================================
# CHART COLORS - Consistent across all charts
# ============================================================================
COLOR_ACTUAL = '#4A3A6B'      # Dark purple for actual/historical prices
COLOR_FORECAST = '#F5C9A8'    # Salmon/Peach for forecast (Hinds Analytics logo)
COLOR_CI_BOOTSTRAP = 'rgba(80, 200, 120, 0.9)'   # Green for Bootstrap CI
COLOR_CI_GARCH = 'rgba(245, 160, 90, 0.9)'       # Orange for GARCH CI
COLOR_CI_FILL = 'rgba(80, 200, 120, 0.2)'        # Light green fill
COLOR_SALMON = '#F5C9A8'      # Salmon/Peach (Hinds Analytics logo)

# ============================================================================
# REGIME CONFIGURATION - Minimum data requirements
# ============================================================================
# ============================================================================
# CUSTOM CSS (identico a la version XGBoost: misma identidad visual)
# ============================================================================
st.markdown("""
<style>
    /* Main metrics cards */
    div[data-testid="metric-container"] {
        background-color: #5A5A5A;
        border: 1px solid #707070;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    div[data-testid="metric-container"] > label {
        color: #E0E0E0 !important;
        font-size: 14px;
    }
    
    div[data-testid="metric-container"] > div {
        color: #FFFFFF !important;
        font-size: 24px;
    }
    
    h1, h2, h3 { color: #FFFFFF !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 14px; }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #5A5A5A;
        border-radius: 6px;
        padding: 14px 34px;          /* mas aire: los titulos no rozan el borde */
        color: #E0E0E0;
        font-size: 16px !important;
        font-weight: 600;
        letter-spacing: .2px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #F5A05A !important;
        color: #FFFFFF !important;
    }
    
    /* Larger subheaders in tabs */
    .stTabs h3 {
        font-size: 24px !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #525252;
    }
    
    .stAlert {
        background-color: #5A5A5A;
        border: 1px solid #707070;
        color: #FFFFFF;
    }
    
    .main-header {
        padding: 10px 0;
        margin-bottom: 10px;
    }
    
    /* AI Assistant column styling */
    .ai-assistant-box {
        background-color: #5A5A5A;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #707070;
        height: 100%;
    }
    
    .chat-message {
        background-color: #656565;
        padding: 10px;
        border-radius: 6px;
        margin: 8px 0;
    }
    
    /* Purple tooltips - ONLY for (?) help icons - Hinds Analytics purple */
    div[data-baseweb="tooltip"] > div {
        background-color: #8B7BB5 !important;
        color: #FFFFFF !important;
    }
    
    /* Data source info */
    .data-source-info {
        background-color: #3A3A3A;
        padding: 10px;
        border-radius: 6px;
        font-size: 12px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_utc_today():
    """Get current date in UTC timezone."""
    return datetime.now(timezone.utc).date()

#def get_utc_midnight_timestamp():
#    """Get timestamp for 00:00 UTC today."""
#    utc_now = datetime.now(timezone.utc)
#    utc_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
#    return int(utc_midnight.timestamp())
def get_utc_midnight_timestamp():
    """Get timestamp for 00:00 UTC yesterday (last closed candle)."""
    utc_now = datetime.now(timezone.utc)
    utc_yesterday_midnight = utc_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    return int(utc_yesterday_midnight.timestamp())

def calculate_adaptive_split(n_samples):
    """
    Calculate train/validation split based on dataset size.
    
    Adaptive approach from Validation Tool V2:
    - Small datasets (<100 samples): 80/20 split (more training to reduce overfitting)
    - Medium datasets (100-200 samples): 75/25 split
    - Large datasets (>200 samples): 70/30 split (more validation for better evaluation)
    
    Returns:
        tuple: (train_size, val_size, train_pct, category)
    """
    MIN_TRAIN = 30   # Absolute minimum for training
    MIN_VAL = 14     # Absolute minimum for validation (2 weeks)
    
    if n_samples < 100:
        train_pct = 0.80
        category = 'Small (<100)'
    elif n_samples < 200:
        train_pct = 0.75
        category = 'Medium (100-200)'
    else:
        train_pct = 0.70
        category = 'Large (>200)'
    
    train_size = int(n_samples * train_pct)
    val_size = n_samples - train_size
    
    # Ensure minimum sizes
    if train_size < MIN_TRAIN:
        train_size = MIN_TRAIN
        val_size = n_samples - train_size
    
    if val_size < MIN_VAL:
        val_size = MIN_VAL
        train_size = n_samples - val_size
    
    return train_size, val_size, train_pct, category

PRICE_CACHE_FILE = "btc_ohlcv_cache.csv"  # mismo cache que escribe collect_data.py


def _get_cc_api_key():
    """Key SOLO desde secrets/env/cc_key.txt (gitignored). Nunca hardcodeada."""
    try:
        key = st.secrets.get("cryptocompare", {}).get("api_key", "")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("CRYPTOCOMPARE_API_KEY", "")
    if key:
        return key
    try:
        if os.path.exists("cc_key.txt"):
            with open("cc_key.txt") as f:
                return f.read().strip()
    except Exception:
        pass
    return ""


def _fetch_cryptocompare(toTs_param):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": "BTC", "tsym": "USD", "limit": 2000, "toTs": toTs_param}
    api_key = _get_cc_api_key()
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare respondio: {data.get('Message', 'sin detalle')}")
    hist = pd.DataFrame(data["Data"]["Data"])
    hist['timestamp'] = pd.to_datetime(hist['time'], unit='s')
    hist = hist[['timestamp', 'close', 'high', 'low']].set_index('timestamp')
    return hist.astype(float)


def _fetch_binance_klines(symbol="BTCUSDT"):
    """Binance klines diarios, sin API key (patron probado en notebook V4.3.3)."""
    rows, end = [], None
    for _ in range(3):                       # 3 x 500 = 1500 velas max
        p = {'symbol': symbol, 'interval': '1d', 'limit': 500}
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
    d = d.set_index('timestamp')[['close', 'high', 'low']].astype(float)
    return d[~d.index.duplicated(keep='last')].sort_index()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_btc_data(toTs_param):
    """BTC OHLC — jerarquia V5: CryptoCompare -> Binance klines -> cache local.

    Solo velas CERRADAS (hasta ayer UTC). Cacheado por dia UTC (ttl=24h).
    Devuelve (hist, source) para mostrar en la UI que fuente respondio.
    """
    hist, source, errors = None, None, []
    try:
        hist = _fetch_cryptocompare(toTs_param)
        source = "CryptoCompare"
    except Exception as e:
        errors.append(f"CryptoCompare: {type(e).__name__}")
        try:
            hist = _fetch_binance_klines("BTCUSDT")
            source = "Binance (klines)"
        except Exception as e2:
            errors.append(f"Binance: {type(e2).__name__}")
            if os.path.exists(PRICE_CACHE_FILE):
                cached = pd.read_csv(PRICE_CACHE_FILE, parse_dates=[0], index_col=0)
                cols = [c for c in ['close', 'high', 'low'] if c in cached.columns]
                hist = cached[cols].astype(float)
                if 'high' not in hist.columns:
                    hist['high'] = hist['close']
                if 'low' not in hist.columns:
                    hist['low'] = hist['close']
                source = f"cache local ({PRICE_CACHE_FILE})"

    if hist is None or len(hist) == 0:
        raise RuntimeError(
            "Sin datos de precio: fallaron todas las fuentes "
            f"({'; '.join(errors)}) y no hay cache local."
        )

    hist = hist.sort_index()
    if source in ("CryptoCompare", "Binance (klines)"):
        try:
            hist.to_csv(PRICE_CACHE_FILE)
        except Exception:
            pass

    yesterday_utc = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    hist = hist[hist.index.date <= yesterday_utc]
    return hist.tail(1000), source

def conditional_sigma(vol_series):
    """Volatilidad condicional saneada: sin ceros ni valores absurdamente bajos.

    Dividir por una sigma cercana a cero dispararia la banda; el piso en el
    percentil 10 acota ese riesgo sin aplanar la senal de volatilidad."""
    s = np.asarray(vol_series, dtype=float)
    finite = s[np.isfinite(s) & (s > 0)]
    if len(finite) == 0:
        return np.ones_like(s)
    floor = max(np.quantile(finite, VOL_FLOOR_Q), 1e-6)
    s = np.where(np.isfinite(s), s, np.median(finite))
    return np.maximum(s, floor)


def normalized_conformal_quantiles(residuals, sigmas, levels=(0.025, 0.975)):
    """Cuantiles de los residuos NORMALIZADOS por volatilidad (z = e/sigma).

    Devuelve cuantiles en unidades de sigma: para obtener la banda de un dia
    concreto se multiplican por la sigma de ESE dia. Asi la banda se adapta al
    regimen en vez de aplicar un ancho fijo a todos los dias."""
    z = np.asarray(residuals, dtype=float) / conditional_sigma(sigmas)
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return None
    return tuple(float(q) for q in np.quantile(z, levels))



# ============================================================================
# MOTOR HAR-RV — el corazon de esta version
# ----------------------------------------------------------------------------
# Reemplaza a XGBoost tras 12 experimentos: a barra diaria, HAR-RV gano en QLIKE
# (0,4978 vs 0,5206) y con potencia aumentada (n_test=3.000 a 8h) la ventaja de
# XGBoost no fue significativa (p=0,31). Cuatro parametros, sin hiperparametros
# que ajustar, sin GridSearch, sin sobreajuste posible.
#
#   RV_{t+1} = c + b_d*RV_t + b_w*RV_semana + b_m*RV_mes
#
# Corsi (2009). Captura el hecho estilizado central: la volatilidad se agrupa y
# tiene memoria larga, con operadores de horizontes distintos superpuestos.
# ============================================================================
EPS = 1e-8


def parkinson_rv(hist):
    """Volatilidad diaria por rango alto-bajo (Parkinson 1980).

    Mucho mas eficiente que |retorno|: usa el recorrido intradia completo en vez
    de solo el cierre. Medido sobre BTC real, su ruido relativo es ~0,53 frente
    a ~0,89 del retorno absoluto.
    """
    hl = np.log(hist['high'] / hist['low'])
    return np.sqrt(hl ** 2 / (4 * np.log(2)))


def build_har_dataset(hist):
    """Componentes diario / semanal / mensual de volatilidad realizada.

    Todo lleva shift(1): la fila del dia t solo contiene informacion hasta t-1.
    Es la misma disciplina que corrigio el leakage de la version anterior.
    """
    d = pd.DataFrame(index=hist.index)
    d['ret'] = np.log(hist['close']).diff()
    d['rv'] = parkinson_rv(hist)
    d['rv_d'] = d['rv'].shift(1)
    d['rv_w'] = d['rv'].rolling(5).mean().shift(1)
    d['rv_m'] = d['rv'].rolling(22).mean().shift(1)
    d['y'] = d['rv']                      # objetivo: la RV del propio dia t
    return d.dropna()


def fit_har(d_train):
    """Minimos cuadrados sobre los tres componentes. Cuatro parametros."""
    A = np.c_[np.ones(len(d_train)), d_train[['rv_d', 'rv_w', 'rv_m']].values]
    coef, *_ = np.linalg.lstsq(A, d_train['y'].values, rcond=None)
    return coef


def predict_har(coef, d_rows):
    A = np.c_[np.ones(len(d_rows)), d_rows[['rv_d', 'rv_w', 'rv_m']].values]
    return np.maximum(A @ coef, EPS)


def har_forecast_path(coef, d, days):
    """Proyeccion recursiva de la volatilidad a `days` dias.

    En cada paso la RV predicha se anexa a la serie y los tres componentes se
    RECALCULAN desde ella. No se desplazan vectores de features: ese fue el bug
    que corrompia la recursion en la version anterior.
    """
    rv_hist = list(d['rv'].values)
    out = []
    for _ in range(days):
        fila = pd.DataFrame([{
            'rv_d': rv_hist[-1],
            'rv_w': float(np.mean(rv_hist[-5:])),
            'rv_m': float(np.mean(rv_hist[-22:])),
        }])
        p = float(predict_har(coef, fila)[0])
        out.append(p)
        rv_hist.append(p)
    return np.array(out)


def har_band_paths(coef, d, days, resid_z, last_log_price, drift=0.0,
                   n_simulations=1000, seed=42):
    """Bandas por simulacion, con volatilidad HAR y shocks normalizados.

    Cada camino: retorno_t = drift + z_t * sigma_HAR(t), donde z son residuos
    estandarizados remuestreados de la historia (conserva colas gruesas reales)
    y sigma evoluciona segun la proyeccion HAR. Devuelve piso, techo, mediana y
    una muestra de caminos para graficar.
    """
    rng = np.random.default_rng(seed)
    sig = har_forecast_path(coef, d, days)
    z = rng.choice(np.asarray(resid_z, dtype=float), size=(n_simulations, days))
    ret = drift + z * sig[None, :]
    precios = np.exp(float(last_log_price) + np.cumsum(ret, axis=1))
    n_show = min(60, n_simulations)
    idx = rng.choice(n_simulations, size=n_show, replace=False)
    return (np.percentile(precios, 2.5, axis=0),
            np.percentile(precios, 97.5, axis=0),
            np.median(precios, axis=0),
            precios[idx], sig)


def baselines_vol(d_train, d_test):
    """Baselines obligatorios: persistencia y EWMA (RiskMetrics)."""
    pers = d_test['rv_w'].values
    lam, ew = 0.94, []
    s = float(d_train['rv'].iloc[:22].mean())
    for v in pd.concat([d_train['rv'], d_test['rv']]).values:
        ew.append(s)
        s = float(np.sqrt(lam * s**2 + (1 - lam) * v**2))
    ew = np.array(ew[-len(d_test):])
    return {'persistencia': pers, 'ewma': ew}


def qlike(y, f):
    """Perdida robusta al ruido del proxy; penaliza mas SUBESTIMAR volatilidad,
    que es el error caro cuando la banda es un presupuesto de riesgo."""
    y2, f2 = np.maximum(y, EPS)**2, np.maximum(f, EPS)**2
    return float(np.mean(y2/f2 - np.log(y2/f2) - 1))

# ============================================================================
# HEADER
# ============================================================================
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown("""
    <div style='padding: 0; margin-bottom: 0;'>
        <p style='margin: 0 0 4px 0; font-size: 56px; font-weight: 780; color: #F5A05A;
                  line-height: 1.05;'>XGBit</p>
        <p style='margin: 0 0 14px 0; font-size: 26px; font-weight: 600; color: #F5C9A8;
                  letter-spacing: .5px;'>Volatility Radar App</p>
        <div style='font-size: 18px; line-height: 1.75; color: #E0E0E0;'>
            <p style='margin: 0;'>▸ Anticipa cuánto puede moverse el Bitcoin</p>
            <p style='margin: 0;'>▸ Encuentra los rangos probables y regímenes del mercado</p>
            <p style='margin: 0;'>▸ Determina cuánto arriesgar en cada posición</p>
        </div>
        <p style='margin: 12px 0 0 0; font-size: 15px; color: #BFC7D0;'>
            <em>Powered by Hinds Analytics</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    _logo_app = None
    for _p in ("assets/XGBit_Logo.JPG", "assets/XGBit_Logo.jpg", "assets/XGBit_Logo.png"):
        if os.path.exists(_p):
            _logo_app = _p
            break
    if _logo_app:
        st.image(_logo_app, width=180)

# ---------------------------------------------------------------------------
# Narrativa en desplegable: la informacion densa no debe abrumar de entrada,
# pero tiene que estar disponible para quien la busque (patron de la ficha
# tecnica plegable). Sintetiza motor + limitaciones en un solo lugar.
# ---------------------------------------------------------------------------
with st.expander("🎯  Qué mide esta app y por qué puedes confiar en el número", expanded=False):
    st.markdown("""
<div style='background-color: #8B7BB5; padding: 18px 22px; border-radius: 8px;'>

**El precio de mañana no se puede predecir. Cuánto puede moverse, sí.**

Esa distinción es todo el producto. XGBit no adivina hacia dónde va Bitcoin: calcula
**hasta dónde puede llegar** y con qué probabilidad, que es la información que
realmente permite dimensionar una posición y colocar un stop donde no lo barra el ruido.

**El motor: memoria de la volatilidad a tres escalas**

Bajo el capó hay un modelo *heterogéneo de volatilidad realizada*. Parte de un hecho
observable en todos los mercados: la volatilidad se contagia entre operadores de
horizontes distintos. Lo que hizo el intradiario ayer, lo que hizo el swing la semana
pasada y lo que hizo el institucional el último mes dejan huellas diferentes, y las tres
juntas explican la agitación de mañana mejor que cualquiera por separado.

El modelo combina esas tres memorias — **día, semana y mes** — en una estimación de
cuánta energía traerá la sesión siguiente. De ahí sale el ancho de la banda: se estrecha
cuando el mercado se calma y se abre cuando se tensa, sin que nadie mueva una perilla.

**Por qué esta arquitectura y no otra**

Llegamos aquí después de doce experimentos con protocolo anti-sesgo: holdout bloqueado,
detección automática de fuga de información, validación walk-forward con costos reales y
comparación obligatoria contra alternativas más simples. Se probaron modelos de machine
learning con búsqueda de hiperparámetros, calibración de probabilidades y horizontes
intradiarios.

El veredicto fue nítido: para **dirección**, ningún modelo superó a la referencia trivial.
Para **volatilidad**, esta arquitectura ganó — y lo hizo con cuatro parámetros, sin nada
que ajustar a mano y sin espacio para sobreajustar. Esa experimentación es la razón por la
que hoy sabes exactamente qué esperar de cada número en pantalla.

**La diferencia que puedes verificar**

Cada banda emitida se registra con fecha inmutable y se compara al día siguiente contra
el cierre real. La pestaña *Track Record* publica ese historial. No prometemos precisión:
la medimos y la enseñamos, acierte o falle.

</div>
    """, unsafe_allow_html=True)

st.caption("⚠️ Herramienta de análisis cuantitativo con fines informativos. No constituye "
           "asesoría financiera ni recomendación de compra o venta. El mercado cripto "
           "conlleva riesgo sustancial de pérdida.")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    _logo_inst = None
    for _p in ("assets/LOGO HindsAnalytics v2026.png", "assets/logo_hinds.png",
               "assets/LOGO_HindsAnalytics.png"):
        if os.path.exists(_p):
            _logo_inst = _p
            break
    if _logo_inst:
        _l1, _l2, _l3 = st.columns([1, 3, 1])
        with _l2:
            st.image(_logo_inst, width=140)
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    st.header("⚙️ Configuración")

    days = st.slider("Días a proyectar", 1, 14, 7,
                     help="Horizonte de las bandas. El ancho crece con el horizonte: "
                          "es incertidumbre acumulada, no una predicción puntual.")

    regime_option = st.selectbox(
        "Régimen aplicado", ["Actual (último)", "Actuales (últimos 2)", "Todo el historial"], index=1,
        help="Sobre qué tramo de historia se ajusta el modelo. Menos historia se adapta más "
             "rápido a un cambio de régimen; más historia estima con menos ruido.")

    show_technical = st.checkbox(
        "Mostrar contexto técnico", value=True,
        help="Panel descriptivo con MA20/50/200, RSI, MACD y ATR. Son observaciones de "
             "contexto, nunca señales de compra o venta.")

    show_paths = st.checkbox(
        "Mostrar caminos simulados", value=True,
        help="Dibuja una muestra de las 1.000 trayectorias que generan la banda.")

    st.divider()
    run_forecast = st.button("🚀 Generar bandas", type="primary", use_container_width=True)
    if st.button("🗑️ Limpiar caché", use_container_width=True):
        st.cache_data.clear()
        st.success("Caché limpiado")

    st.markdown("""
    <div class='data-source-info'>
        <strong>📡 Fuente:</strong> CryptoCompare → Binance → cache<br>
        <strong>🧠 Motor:</strong> HAR-RV (Corsi, 2009)<br>
        <strong>🕐 Timezone:</strong> UTC · solo velas cerradas
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Sobre esta versión"):
        st.markdown("""
        **Por qué HAR-RV y no XGBoost.** Doce experimentos con holdout bloqueado,
        test anti-leakage y baselines obligatorios concluyeron que el machine
        learning no aportaba: peor que el pronóstico ingenuo en dirección
        (Theil U2 = 1,046) y sin ventaja significativa en volatilidad (p = 0,31
        con 3.000 observaciones de test).

        **Qué mide esta app:** el rango probable del precio, con cobertura
        empírica publicada; el régimen vigente y su deriva; y el contexto técnico
        de forma descriptiva.

        **Qué NO hace:** no predice dirección ni emite señales de compra o venta.
        """)

    try:
        if authenticator.logout("🚪 Salir", "sidebar"):
            st.session_state.clear()
            st.rerun()
    except Exception:
        pass


# ============================================================================
# ASISTENTE AI
# ----------------------------------------------------------------------------
# El prompt se reescribio por completo respecto de la version XGBoost: aquel
# giraba en torno a la P direccional y al calculo de EV con esa probabilidad,
# que en esta version NO existe (resolucion medida 0,0002, AUC 0,503 -> la P no
# discriminaba y se retiro). Aqui el asistente explica bandas, volatilidad,
# regimenes y dimensionamiento, y tiene prohibido dar direccion.
# ============================================================================
def construir_contexto():
    """Arma el estado actual de la app para el prompt del asistente."""
    fd = st.session_state.get('forecast_df')
    cov = st.session_state.get('band_coverage', {})
    cal = st.session_state.get('calidad')
    coef = st.session_state.get('har_coef')
    sig = st.session_state.get('sig_path')
    rs = st.session_state.get('regime_stats')
    lp = st.session_state.get('last_price')
    if fd is None:
        return "Todavia no se han generado bandas en esta sesion."

    lo1, hi1 = fd['Piso 95%'].iloc[0], fd['Techo 95%'].iloc[0]
    semi = (hi1 - lo1) / 2 / lp * 100
    partes = [
        f"Precio actual (ultimo cierre): ${lp:,.2f}",
        f"Volatilidad esperada manana: {sig[0]*100:.2f}% diario",
        f"Banda 95% dia 1: ${lo1:,.0f} - ${hi1:,.0f} (semiancho +-{semi:.2f}%)",
        f"Banda 95% dia {len(fd)}: ${fd['Piso 95%'].iloc[-1]:,.0f} - ${fd['Techo 95%'].iloc[-1]:,.0f}",
        f"Cobertura empirica medida: banda 95% -> {cov.get('coverage_pct', float('nan')):.1f}%, "
        f"banda 68% -> {cov.get('coverage_68_pct', float('nan')):.1f}% "
        f"(sobre {cov.get('n_days', 0)} dias out-of-sample)",
    ]
    if coef is not None:
        partes.append(f"Ecuacion HAR ajustada: RV(t+1) = {coef[0]:.5f} + {coef[1]:.3f}*RV_dia "
                      f"+ {coef[2]:.3f}*RV_semana + {coef[3]:.3f}*RV_mes")
    if cal is not None and len(cal):
        partes.append("Comparacion QLIKE (menor es mejor): " +
                      " | ".join(f"{r['modelo']}={r['QLIKE']}" for _, r in cal.iterrows()))
    if rs is not None and len(rs):
        u = rs.iloc[-1]
        partes.append(f"Regimen vigente: {int(u['days'])} dias de antiguedad, "
                      f"volatilidad {u['volatility_%']}%/dia, deriva acumulada {u['total_return_%']}%")
    return "\n".join(partes)


PROMPT_SISTEMA = """Eres el asistente de BTC Risk Bands, una herramienta de analisis
cuantitativo de Bitcoin. Tu funcion es explicar BANDAS DE RIESGO, VOLATILIDAD,
REGIMENES y DIMENSIONAMIENTO DE POSICION. No eres asesor financiero ni generas senales.

QUE ES ESTE PRODUCTO (y que NO es):
- Publica el RANGO probable del precio, con la cobertura de ese rango medida y verificable.
- El motor es HAR-RV (Corsi, 2009): RV(t+1) = c + b_d*RV_dia + b_w*RV_semana + b_m*RV_mes.
  Cuatro parametros por minimos cuadrados. No hay machine learning.
- NO pronostica direccion, y esto no es modestia: doce experimentos con protocolo
  anti-sesgo lo establecieron. El Theil U2 fue 1,046 con IC95 [1,025-1,080], es decir
  PEOR que el pronostico ingenuo "manana = hoy", con Diebold-Mariano significativo.
  La probabilidad direccional que existia en la version anterior se retiro porque su
  resolucion medida fue 0,0002 y su AUC 0,503: no discriminaba nada.
- XGBoost fue eliminado tras perder frente a HAR-RV en QLIKE y no mostrar ventaja
  significativa ni con 3.000 observaciones de test (p=0,31).

COMO EXPLICAR EL DIMENSIONAMIENTO (es tu funcion principal):
  tamano_posicion = riesgo_maximo / semiancho_de_banda
Ejemplo: riesgo maximo 2% del capital y semiancho +-4% -> 0,5x el tamano estandar.
Cuando la volatilidad esperada sube, la banda se ensancha y el tamano debe bajar en
proporcion. Muestra el calculo con los numeros concretos que te de el usuario.

REGLAS QUE PUEDES ENSENAR:
1. Banda = presupuesto de riesgo -> dimensiona la posicion.
2. Stops FUERA de la banda 95%; targets DENTRO. Un stop dentro de la banda es ruido
   esperado y te sacara sin que ocurra nada informativo.
3. La volatilidad esperada indica cuanto encoger o ampliar la exposicion.
4. Esta herramienta no da direccion: si el usuario opera, la tesis direccional debe
   venir de otro lado; esto dimensiona el riesgo de esa tesis.
5. Cambio de regimen detectado = revisar posiciones, no abrir tamano nuevo.

COMO INTERPRETAR LA COBERTURA:
- La banda 95% deberia contener el cierre real ~95% de los dias, y la de 68% un ~68%.
- Si el 95% cubre pero el 68% no, la forma de la distribucion esta mal aunque las colas
  acierten. Por eso se publican dos niveles.
- Si la cobertura esta por debajo de lo nominal, el usuario debe ampliar el semiancho al
  dimensionar (por ejemplo multiplicar por 1,2) hasta que la cobertura viva lo confirme.

ESTADO ACTUAL DE LA APP:
{contexto}

PROHIBIDO (reglas duras, sin excepciones):
1. NO recomendar direccion: nunca digas comprar, vender, ir largo o corto, ni sugieras
   puntos de entrada o salida. Si te lo piden, reencuadra hacia el dimensionamiento y la
   colocacion de stops, y recuerda que la decision direccional es del usuario.
2. NO afirmar rendimiento superior, alta precision, ni que la herramienta "predice" el
   precio. Si preguntan que tan buena es, cita la cobertura empirica y el QLIKE frente a
   los baselines.
3. NO inventes metricas ni datos que no esten arriba. Si no lo sabes, dilo.
4. NO presentes la mediana de los caminos como un objetivo de precio.
5. Si el usuario pregunta por otra configuracion, pidele que genere bandas con ella.
6. Menciona la naturaleza informativa (no asesoria financiera) cuando la conversacion se
   acerque a una decision de inversion.

ESTILO: conciso, con los numeros reales de arriba, en el idioma del usuario."""


def render_asistente():
    st.markdown("### 🤖 Asistente")
    st.caption("Explica bandas, volatilidad y dimensionamiento. No da dirección.")

    if 'mensajes' not in st.session_state:
        st.session_state.mensajes = []

    contenedor = st.container(height=380)
    with contenedor:
        for m in st.session_state.mensajes:
            with st.chat_message(m['role']):
                st.markdown(m['content'])

    pregunta = st.chat_input("Pregunta sobre las bandas...")

    st.caption("**Preguntas rápidas**")
    rapidas = [
        ("📏 ¿Cómo dimensiono?", "Con el semiancho de la banda del día 1, ¿cómo calculo el "
                                 "tamaño de posición si mi riesgo máximo por operación es 2% del capital?"),
        ("🎯 ¿Dónde pongo el stop?", "Según la regla de stops fuera de la banda 95%, ¿qué niveles "
                                     "se desprenden de las bandas de hoy?"),
        ("📊 ¿Son confiables las bandas?", "¿Qué significa la cobertura medida y cómo debo "
                                           "interpretarla al dimensionar?"),
        ("⚠️ ¿Qué riesgos hay hoy?", "¿Cuáles son los principales riesgos según el régimen "
                                     "actual y la volatilidad esperada?"),
    ]
    for etiqueta, texto in rapidas:
        if st.button(etiqueta, use_container_width=True, key=f"q_{etiqueta}"):
            pregunta = texto

    if pregunta:
        st.session_state.mensajes.append({'role': 'user', 'content': pregunta})
        try:
            from groq import Groq
            api_key = ""
            try:
                api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
            if not api_key:
                api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                st.error("⚠️ Falta GROQ_API_KEY en los secrets.")
                st.stop()

            cliente = Groq(api_key=api_key)
            sistema = PROMPT_SISTEMA.format(contexto=construir_contexto())
            resp = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": sistema}] +
                         [{"role": m['role'], "content": m['content']}
                          for m in st.session_state.mensajes[-8:]],
                temperature=0.3, max_tokens=800)
            texto = resp.choices[0].message.content
            st.session_state.mensajes.append({'role': 'assistant', 'content': texto})
            st.rerun()
        except Exception as e:
            st.error(f"Error del asistente: {type(e).__name__}: {e}")

    if st.session_state.mensajes and st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

    st.caption("*Powered by Groq (Llama 3.3)*")


# ============================================================================
# MAIN — se ejecuta al pulsar el botón
# ============================================================================
if run_forecast:
    try:
        # --- 1. datos -------------------------------------------------------
        with st.spinner("📡 Descargando precios..."):
            hist, data_source = fetch_btc_data(get_utc_midnight_timestamp())
            st.session_state['hist'] = hist
            st.session_state['data_source'] = data_source
        st.success(f"✅ {len(hist)} velas · fuente: {data_source} · "
                   f"último cierre {hist.index[-1].date()}")

        # --- 2. regímenes ---------------------------------------------------
        with st.spinner("🔍 Detectando regímenes..."):
            serie = hist['close'].values
            algo = rpt.Pelt(model='rbf', min_size=30).fit(serie)
            breakpoints = algo.predict(pen=10)
            st.session_state['breakpoints'] = breakpoints

            filas, prev = [], 0
            for i, bp in enumerate(breakpoints):
                seg = hist['close'].iloc[prev:bp]
                if len(seg) > 2:
                    filas.append({
                        'regimen': i + 1,
                        'inicio': str(seg.index[0].date()),
                        'fin': str(seg.index[-1].date()),
                        'days': len(seg),
                        'volatility_%': round(seg.pct_change().std() * 100, 2),
                        'total_return_%': round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1),
                    })
                prev = bp
            regime_stats = pd.DataFrame(filas)
            st.session_state['regime_stats'] = regime_stats

        # tramo de entrenamiento según el régimen elegido
        if regime_option == "Actual (último)" and len(breakpoints) >= 2:
            ini = breakpoints[-2]
        elif regime_option == "Actuales (últimos 2)" and len(breakpoints) >= 3:
            ini = breakpoints[-3]
        else:
            ini = 0
        hist_regime = hist.iloc[max(0, ini):]
        if len(hist_regime) < 150:                       # HAR necesita ~22 + margen
            hist_regime = hist.iloc[-max(250, len(hist_regime)):]
        st.session_state['hist_regime'] = hist_regime
        regime_label = f"{len(hist_regime)} días de entrenamiento"

        # --- 3. HAR-RV ------------------------------------------------------
        with st.spinner("📐 Ajustando HAR-RV..."):
            d = build_har_dataset(hist_regime)
            cut = int(len(d) * 0.75)
            d_tr, d_va = d.iloc[:cut], d.iloc[cut:]

            coef = fit_har(d_tr)
            pred_va = predict_har(coef, d_va)
            real_va = d_va['y'].values

            base = baselines_vol(d_tr, d_va)
            calidad = pd.DataFrame(
                [{'modelo': 'HAR-RV', 'QLIKE': round(qlike(real_va, pred_va), 4),
                  'R2': round(float(np.corrcoef(pred_va, real_va)[0, 1] ** 2), 4)}] +
                [{'modelo': k, 'QLIKE': round(qlike(real_va, v), 4),
                  'R2': round(float(np.corrcoef(v, real_va)[0, 1] ** 2), 4)}
                 for k, v in base.items()]
            ).sort_values('QLIKE').reset_index(drop=True)
            st.session_state['calidad'] = calidad
            st.session_state['har_coef'] = coef

            # residuos estandarizados: retorno / volatilidad esperada de ese día
            sig_tr = predict_har(coef, d_tr)
            z_tr = d_tr['ret'].values / np.maximum(sig_tr, EPS)
            z_tr = z_tr[np.isfinite(z_tr)]
            sig_va = pred_va
            z_va = d_va['ret'].values / np.maximum(sig_va, EPS)
            z_va = z_va[np.isfinite(z_va)]

            # cobertura empírica OOS (conformal: cuantiles de la 1ª mitad, evaluación en la 2ª)
            h = len(z_va) // 2
            q95 = np.quantile(z_va[:h], [.025, .975])
            q68 = np.quantile(z_va[:h], [.16, .84])
            ev_z, ev_sig = z_va[h:], sig_va[h:]
            ev_ret = ev_z * ev_sig
            cob95 = float(np.mean((ev_ret >= q95[0] * ev_sig) & (ev_ret <= q95[1] * ev_sig))) * 100
            cob68 = float(np.mean((ev_ret >= q68[0] * ev_sig) & (ev_ret <= q68[1] * ev_sig))) * 100
            st.session_state['band_coverage'] = {
                'coverage_pct': cob95, 'coverage_68_pct': cob68,
                'n_days': int(len(ev_ret)), 'q95': q95.tolist(), 'q68': q68.tolist()}

        # --- 4. bandas ------------------------------------------------------
        with st.spinner("🎲 Simulando 1.000 caminos..."):
            drift = float(d['ret'].tail(252).mean())     # deriva histórica, no pronóstico
            last_price = float(hist_regime['close'].iloc[-1])
            lo, hi, med, paths, sig_path = har_band_paths(
                coef, d, days, z_tr, np.log(last_price), drift=drift, n_simulations=1000)

            fechas = pd.date_range(hist_regime.index[-1] + timedelta(days=1), periods=days, freq='D')
            forecast_df = pd.DataFrame({
                'Fecha': fechas, 'Referencia': med, 'Piso 95%': lo, 'Techo 95%': hi,
                'Ancho %': (hi - lo) / med * 100,
                'Vol esperada %': sig_path * 100,
            })
            st.session_state.update({'forecast_df': forecast_df, 'sample_paths': paths,
                                     'last_price': last_price, 'days': days,
                                     'regime_label': regime_label, 'sig_path': sig_path,
                                     'val_dates': d_va.index, 'val_real': real_va,
                                     'val_pred': pred_va, 'd_full': d})
        st.session_state['listo'] = True

    except Exception as e:
        st.error(f"❌ Error: {type(e).__name__}: {e}")
        with st.expander("Detalle"):
            import traceback
            st.code(traceback.format_exc())
        st.session_state['listo'] = False

# ============================================================================
# TABS
# ============================================================================
if st.session_state.get('listo'):
    forecast_df = st.session_state['forecast_df']
    last_price = st.session_state['last_price']
    days = st.session_state['days']
    hist = st.session_state['hist']
    hist_regime = st.session_state['hist_regime']
    _cov = st.session_state['band_coverage']
    sig_path = st.session_state['sig_path']

    col_main, col_chat = st.columns([4, 0.8])

    with col_chat:
        render_asistente()

    with col_main:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["◈  Bandas", "◧  Regímenes", "◎  Calidad del modelo", "◷  Track Record"])

        # ------------------------------------------------------------------ TAB 1
        with tab1:
            st.subheader(f"Proyección a {days} días · {st.session_state['regime_label']}")

            lo1, hi1 = forecast_df['Piso 95%'].iloc[0], forecast_df['Techo 95%'].iloc[0]
            loN, hiN = forecast_df['Piso 95%'].iloc[-1], forecast_df['Techo 95%'].iloc[-1]
            semi1 = (hi1 - lo1) / 2 / last_price * 100

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Precio actual", f"${last_price:,.0f}",
                          help="Último cierre diario confirmado.")
            with c2:
                st.metric("Vol esperada mañana", f"±{sig_path[0]*100:.2f}%",
                          help="Volatilidad diaria que proyecta el HAR-RV para mañana. "
                               "Es lo que este modelo sí sabe estimar.")
            with c3:
                st.metric("Banda 95% — Día 1", f"${lo1:,.0f} – ${hi1:,.0f}",
                          f"semiancho ±{semi1:.1f}%", delta_color="off",
                          help="Rango donde se espera el cierre de mañana con 95% de confianza.")
            with c4:
                st.metric(f"Banda 95% — Día {days}", f"${loN:,.0f} – ${hiN:,.0f}",
                          f"ancho {(hiN-loN)/forecast_df['Referencia'].iloc[-1]*100:.1f}%",
                          delta_color="off",
                          help="El ancho crece con el horizonte: incertidumbre acumulada.")
            with c5:
                st.metric("Presupuesto de riesgo", f"±{semi1:.1f}%",
                          f"cobertura medida: {_cov['coverage_pct']:.0f}%", delta_color="off",
                          help="Tamaño de posición = riesgo_máximo / semiancho. "
                               "Stops FUERA de la banda; targets DENTRO.")

            # --- gráfico ---
            fig = go.Figure()
            h_show = hist_regime.tail(60)
            fig.add_trace(go.Scatter(x=h_show.index, y=h_show['close'], name='Histórico',
                                     mode='lines', line=dict(color=COLOR_ACTUAL, width=2)))
            if show_paths:
                for i, p in enumerate(st.session_state['sample_paths']):
                    fig.add_trace(go.Scatter(x=forecast_df['Fecha'], y=p, mode='lines',
                                             line=dict(color='rgba(245,160,90,0.13)', width=1),
                                             showlegend=(i == 0), hoverinfo='skip',
                                             name='Caminos simulados (muestra de 1.000)'))
            fig.add_trace(go.Scatter(x=forecast_df['Fecha'], y=forecast_df['Techo 95%'],
                                     name='Techo 95%', mode='lines+markers',
                                     line=dict(color=COLOR_CI_BOOTSTRAP, width=3), marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=forecast_df['Fecha'], y=forecast_df['Piso 95%'],
                                     name='Piso 95%', mode='lines+markers',
                                     line=dict(color=COLOR_CI_BOOTSTRAP, width=3), marker=dict(size=6),
                                     fill='tonexty', fillcolor=COLOR_CI_FILL))
            fig.add_trace(go.Scatter(x=forecast_df['Fecha'], y=forecast_df['Referencia'],
                                     name='Mediana de los caminos', mode='lines',
                                     line=dict(color=COLOR_FORECAST, width=1, dash='dot'), opacity=.5))
            fig.add_vline(x=hist_regime.index[-1], line=dict(color='#CC4444', dash='dash', width=2),
                          annotation_text='Hoy (UTC)')
            fig.update_layout(plot_bgcolor='#808080', paper_bgcolor='#5A5A5A',
                              font=dict(color='#FFFFFF'), hovermode='x unified', height=470,
                              margin=dict(l=60, r=20, t=30, b=50),
                              xaxis=dict(title='Fecha', showgrid=False),
                              yaxis=dict(title='Precio (USD)', showgrid=False, tickformat='$,.0f'),
                              legend=dict(bgcolor='rgba(90,90,90,.8)'))
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Las líneas tenues son una muestra de los 1.000 caminos simulados que generan la "
                "banda: cada uno es un futuro posible con volatilidad realista. La línea punteada es "
                "su mediana, y sale suave por aritmética —los shocks se cancelan al agregar— no "
                "porque el modelo suavice nada. **No es un objetivo de precio:** esta app no "
                "pronostica dirección. El 95% de los caminos queda dentro de la banda.")

            with st.expander("📖 Cómo usar estas bandas (5 reglas)", expanded=False):
                st.markdown("""
    <div style='background-color: #8B7BB5; padding: 15px; border-radius: 8px;'>

    **1 · Banda = presupuesto de riesgo → dimensiona la posición**

    _Tamaño ≈ riesgo_máximo / semiancho. Riesgo máximo 2% y semiancho ±4% → 0,5× tu tamaño estándar._

    **2 · Stops FUERA de la banda 95%; targets DENTRO**

    _Un stop dentro de la banda es ruido esperado: te sacará sin que ocurra nada informativo._

    **3 · La vol esperada te dice cuánto encoger o ampliar**

    _Cuando la vol proyectada sube, la banda se ensancha sola y tu tamaño debe bajar en proporción._

    **4 · Esta herramienta no da dirección**

    _Doce experimentos lo establecieron. Si operas, la tesis direccional debe venir de otro lado;
    esto dimensiona el riesgo de esa tesis._

    **5 · Cambio de régimen detectado = revisar posiciones, no abrir tamaño nuevo**

    _Tras un quiebre estructural, bandas y cobertura se estiman sobre un mercado que ya cambió._

    </div>
                """, unsafe_allow_html=True)

            # --- contexto técnico ---
            _card = ("<div style='background-color:#3A3A3A;padding:12px;border-radius:8px;"
                     "text-align:center;border-top:3px solid {c};min-height:118px;'>"
                     "<p style='margin:0;font-size:12px;color:#999;'>{t}</p>"
                     "<p style='margin:8px 0;font-size:20px;color:{c};font-weight:bold;'>{v}</p>"
                     "<p style='margin:0;font-size:12px;color:#BBB;'>{d}</p></div>")
            _rs = st.session_state.get('regime_stats', pd.DataFrame())
            reg_tipo, reg_dias, reg_deriva = "—", 0, 0.0
            if len(_rs):
                mv = _rs['volatility_%'].median(); ult = _rs.iloc[-1]
                reg_tipo = "estable" if ult['volatility_%'] <= mv else "volátil"
                reg_dias, reg_deriva = int(ult['days']), float(ult['total_return_%'])

            if show_technical:
                st.markdown("---")
                st.markdown("##### 🧭 Contexto Técnico (descriptivo)")
                _c = hist['close']; _px = float(_c.iloc[-1])
                ma20, ma50, ma200 = _c.rolling(20).mean(), _c.rolling(50).mean(), _c.rolling(200).mean()
                def _dist(ma):
                    v = ma.iloc[-1]
                    return None if pd.isna(v) else (_px / float(v) - 1) * 100
                _d = _c.diff()
                _g = _d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                _l = (-_d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                rsi = float((100 - 100 / (1 + _g / _l.replace(0, np.nan))).iloc[-1])
                macd = _c.ewm(span=12, adjust=False).mean() - _c.ewm(span=26, adjust=False).mean()
                sig_l = macd.ewm(span=9, adjust=False).mean()
                above = (macd > sig_l).astype(int)
                cr = np.where((above.diff().fillna(0) != 0).values)[0]
                dias_cruce = int(len(_c) - 1 - cr[-1]) if len(cr) else None
                cprev = _c.shift(1)
                tr = pd.concat([(hist['high'] - hist['low']), (hist['high'] - cprev).abs(),
                                (hist['low'] - cprev).abs()], axis=1).max(axis=1)
                atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
                def _f(x): return "n/d" if x is None else f"{x:+.1f}%"
                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    st.markdown(_card.format(c="#8B7BB5", t="DISTANCIA A MEDIAS",
                                             v=f"{_f(_dist(ma200))} vs MA200",
                                             d=f"MA20: {_f(_dist(ma20))} · MA50: {_f(_dist(ma50))}"),
                                unsafe_allow_html=True)
                with t2:
                    zona = "sobrecompra (>70)" if rsi > 70 else ("sobreventa (<30)" if rsi < 30
                                                                else "zona neutra (30–70)")
                    st.markdown(_card.format(c="#8B7BB5", t="RSI (14)", v=f"{rsi:.0f}", d=zona),
                                unsafe_allow_html=True)
                with t3:
                    st.markdown(_card.format(c="#8B7BB5", t="MACD — DÍAS DESDE EL CRUCE",
                                             v=f"día {dias_cruce}" if dias_cruce is not None else "n/d",
                                             d="MACD sobre su señal" if above.iloc[-1] == 1
                                               else "MACD bajo su señal"), unsafe_allow_html=True)
                with t4:
                    st.markdown(_card.format(c="#8B7BB5", t="ATR (14)", v=f"${atr:,.0f}",
                                             d=f"{atr/_px*100:.1f}% del precio"), unsafe_allow_html=True)
                with st.expander("📉 Ver gráfico con MA20 / MA50 / MA200"):
                    w = hist.tail(365)
                    fma = go.Figure()
                    fma.add_trace(go.Scatter(x=w.index, y=w['close'], name='Precio',
                                             line=dict(color=COLOR_ACTUAL, width=2)))
                    for s_, n_, c_ in ((ma20, 'MA20', '#F5C9A8'), (ma50, 'MA50', '#F5A05A'),
                                       (ma200, 'MA200', '#8B7BB5')):
                        sv = s_.reindex(w.index)
                        if sv.notna().any():
                            fma.add_trace(go.Scatter(x=w.index, y=sv, name=n_,
                                                     line=dict(color=c_, width=1.5)))
                    fma.update_layout(plot_bgcolor='#808080', paper_bgcolor='#5A5A5A',
                                      font=dict(color='#FFF'), height=400, hovermode='x unified',
                                      margin=dict(l=60, r=20, t=20, b=50),
                                      yaxis=dict(tickformat='$,.0f', showgrid=False),
                                      xaxis=dict(showgrid=False))
                    st.plotly_chart(fma, use_container_width=True)

            # --- contexto de mercado ---
            st.markdown("---")
            st.markdown("##### 📊 Contexto de mercado (descriptivo)")
            vol_reg = hist_regime['close'].pct_change().std() * 100
            v1 = sig_path[0] * 100
            etiqueta, color_v = (("baja", "#44FF44") if v1 < vol_reg * .8 else
                                 ("alta", "#FF6B6B") if v1 > vol_reg * 1.25 else ("media", "#F5A05A"))
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(_card.format(c=color_v, t="VOL ESPERADA (DÍA 1)", v=etiqueta,
                                         d=f"{v1:.2f}%/día · régimen {vol_reg:.2f}%/día"),
                            unsafe_allow_html=True)
            with m2:
                st.markdown(_card.format(c="#44FF44" if reg_tipo == "estable" else "#F5A05A",
                                         t="RÉGIMEN ACTUAL", v=reg_tipo,
                                         d=f"{reg_dias} días de antigüedad"), unsafe_allow_html=True)
            with m3:
                st.markdown(_card.format(c="#44FF44" if reg_deriva >= 0 else "#FF6B6B",
                                         t="DERIVA DEL RÉGIMEN", v=f"{reg_deriva:+.1f}%",
                                         d=f"acumulada en {reg_dias} días"), unsafe_allow_html=True)
            with m4:
                st.markdown(_card.format(c="#8B7BB5", t="COBERTURA MEDIDA",
                                         v=f"{_cov['coverage_pct']:.0f}%",
                                         d=f"nominal 95% · {_cov['n_days']} días OOS"),
                            unsafe_allow_html=True)

            st.markdown("""
            <div style='background-color:#4A3A3A;padding:12px;border-radius:8px;
                        border-left:4px solid #CC4444;margin-top:10px;'>
                <p style='margin:0;font-size:13px;color:#D0D0D0;'>
                    ⚠️ <strong>Solo con fines informativos — NO es asesoría financiera.</strong>
                    Todos los paneles son observaciones cuantitativas descriptivas, no recomendaciones.
                    El mercado cripto conlleva riesgo sustancial de pérdida.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### Tabla detallada")
            tdf = forecast_df.copy()
            tdf['Fecha'] = tdf['Fecha'].dt.strftime('%Y-%m-%d')
            for c in ('Referencia', 'Piso 95%', 'Techo 95%'):
                tdf[c] = tdf[c].apply(lambda x: f"${x:,.0f}")
            tdf['Ancho %'] = tdf['Ancho %'].apply(lambda x: f"{x:.1f}%")
            tdf['Vol esperada %'] = tdf['Vol esperada %'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(tdf, use_container_width=True, hide_index=True)

        # ------------------------------------------------------------------ TAB 2
        with tab2:
            st.subheader("Detección de regímenes")
            rs = st.session_state.get('regime_stats', pd.DataFrame())
            if len(rs):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Regímenes detectados", len(rs))
                with c2:
                    st.metric("Antigüedad del actual", f"{int(rs.iloc[-1]['days'])} días")
                with c3:
                    st.metric("Deriva del régimen actual", f"{float(rs.iloc[-1]['total_return_%']):+.1f}%",
                              delta_color="off",
                              help="Retorno acumulado dentro del régimen vigente. Describe lo que YA "
                                   "ocurrió; no es una proyección.")
                figr = go.Figure()
                figr.add_trace(go.Scatter(x=hist.index, y=hist['close'], name='BTC',
                                          line=dict(color=COLOR_ACTUAL, width=1.5)))
                for bp in st.session_state['breakpoints'][:-1]:
                    figr.add_vline(x=hist.index[min(bp, len(hist) - 1)],
                                   line=dict(color='#F5A05A', dash='dash', width=1))
                figr.update_layout(plot_bgcolor='#808080', paper_bgcolor='#5A5A5A',
                                   font=dict(color='#FFF'), height=420, hovermode='x unified',
                                   margin=dict(l=60, r=20, t=20, b=50),
                                   yaxis=dict(tickformat='$,.0f', showgrid=False),
                                   xaxis=dict(showgrid=False))
                st.plotly_chart(figr, use_container_width=True)
                st.dataframe(rs, use_container_width=True, hide_index=True)
                st.caption("Los quiebres se detectan con PELT sobre la serie de precios. "
                           "Un cambio de régimen es motivo para revisar posiciones abiertas, "
                           "no para abrir tamaño nuevo (regla 5).")

        # ------------------------------------------------------------------ TAB 3
        with tab3:
            st.subheader("Calidad del modelo")
            st.caption("La pregunta no es si la app acierta el precio —no lo intenta— sino si el "
                       "RANGO que publica es confiable y si el modelo de volatilidad supera a "
                       "alternativas más simples.")

            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("⭐ Cobertura banda 95%", f"{_cov['coverage_pct']:.0f}%",
                          f"n = {_cov['n_days']} días OOS", delta_color="off",
                          help="Porcentaje de cierres reales dentro de la banda 95% out-of-sample. "
                               "Es LA métrica del producto.")
            with q2:
                st.metric("Cobertura banda 68%", f"{_cov['coverage_68_pct']:.0f}%",
                          "nominal: 68%", delta_color="off",
                          help="Dos niveles dicen más que uno: si el 95% cubre pero el 68% no, "
                               "la forma de la distribución está mal aunque las colas acierten.")
            with q3:
                cal = st.session_state['calidad']
                st.metric("Motor ganador (QLIKE)", cal.iloc[0]['modelo'],
                          f"QLIKE {cal.iloc[0]['QLIKE']:.4f}", delta_color="off",
                          help="QLIKE penaliza más subestimar la volatilidad, que es el error caro "
                               "cuando la banda es un presupuesto de riesgo.")

            c = _cov['coverage_pct']
            if abs(c - 95) <= 4:
                st.success(f"✅ Bandas confiables: {c:.0f}% de los cierres dentro de la banda 95%.")
            elif 88 <= c <= 99:
                st.warning(f"⚠️ Cobertura {c:.0f}%: usable con margen. Amplía el semiancho ~1,2× "
                           "al dimensionar hasta tener cobertura viva.")
            else:
                st.error(f"❌ Cobertura {c:.0f}%: la banda no representa lo que anuncia.")

            st.markdown("##### HAR-RV frente a alternativas más simples")
            st.dataframe(st.session_state['calidad'], use_container_width=True, hide_index=True)
            st.caption("Si la persistencia o el EWMA ganaran de forma sostenida, HAR-RV sobraría — "
                       "el mismo criterio de parsimonia que retiró a XGBoost de esta app.")

            st.markdown("##### Volatilidad estimada vs. realizada (validación)")
            figv = go.Figure()
            figv.add_trace(go.Scatter(x=st.session_state['val_dates'],
                                      y=st.session_state['val_real'] * 100,
                                      name='Volatilidad realizada', mode='lines',
                                      line=dict(color=COLOR_ACTUAL, width=1.5)))
            figv.add_trace(go.Scatter(x=st.session_state['val_dates'],
                                      y=st.session_state['val_pred'] * 100,
                                      name='HAR-RV', mode='lines',
                                      line=dict(color=COLOR_FORECAST, width=2)))
            figv.update_layout(plot_bgcolor='#808080', paper_bgcolor='#5A5A5A',
                               font=dict(color='#FFF'), height=380, hovermode='x unified',
                               margin=dict(l=60, r=20, t=20, b=50),
                               yaxis=dict(title='vol diaria (%)', showgrid=False),
                               xaxis=dict(showgrid=False))
            st.plotly_chart(figv, use_container_width=True)
            st.caption("Aquí la línea SÍ debe seguir a la realidad: la volatilidad, a diferencia de "
                       "la dirección, es predecible. Ese contraste resume por qué este producto "
                       "existe y por qué no promete dirección.")

            coef = st.session_state['har_coef']
            st.markdown("##### El modelo completo, en una ecuación")
            st.latex(r"RV_{t+1} = %.5f + %.3f\,RV_t + %.3f\,\overline{RV}_{5} + %.3f\,\overline{RV}_{22}"
                     % (coef[0], coef[1], coef[2], coef[3]))
            st.caption("Cuatro parámetros, ajustados por mínimos cuadrados. Auditable a simple vista: "
                       "no hay hiperparámetros, ni búsqueda, ni posibilidad de sobreajuste relevante.")

        # ------------------------------------------------------------------ TAB 4
        with tab4:
            st.subheader("Track Record / Model Health")
            st.caption("Lo escribe monitor.py cada madrugada vía GitHub Actions. Cada fila queda con "
                       "un commit de fecha inmutable: el historial del archivo en el repo ES la "
                       "prueba auditable del track record.")
            try:
                hl = pd.read_csv('monitoring/health_log.csv', parse_dates=['fecha']).sort_values('fecha')
            except Exception:
                hl = None
            if hl is None or len(hl) == 0:
                st.info("📭 **El monitor aún no ha escrito historial.**\n\n"
                        "Este tab se llena cuando `monitor.py` corra en GitHub Actions y haga commit "
                        "de `monitoring/health_log.csv`. Hasta entonces, la cobertura que ves en "
                        "'Calidad del modelo' es una estimación out-of-sample sobre la validación, "
                        "no track record en vivo — y así está etiquetada.")
            else:
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Días monitoreados", len(hl))
                with k2:
                    cacc = hl['dentro_banda_95'].mean() * 100 if 'dentro_banda_95' in hl else None
                    st.metric("Cobertura acumulada", f"{cacc:.0f}%" if cacc is not None else "n/d",
                              delta_color="off")
                with k3:
                    c90 = (hl['cobertura_rodante_90d'].dropna().iloc[-1]
                           if 'cobertura_rodante_90d' in hl and hl['cobertura_rodante_90d'].notna().any()
                           else None)
                    st.metric("Cobertura rodante 90d", f"{c90:.0f}%" if c90 is not None else "n/d",
                              "alerta si <88%", delta_color="off")
                p = hl.tail(120)
                figt = go.Figure()
                if {'banda_lo_95', 'banda_hi_95'}.issubset(p.columns):
                    figt.add_trace(go.Scatter(x=p['fecha'], y=p['banda_hi_95'], name='Techo emitido',
                                              line=dict(color=COLOR_CI_BOOTSTRAP, width=2)))
                    figt.add_trace(go.Scatter(x=p['fecha'], y=p['banda_lo_95'], name='Piso emitido',
                                              line=dict(color=COLOR_CI_BOOTSTRAP, width=2),
                                              fill='tonexty', fillcolor=COLOR_CI_FILL))
                figt.add_trace(go.Scatter(x=p['fecha'], y=p['cierre'], name='Cierre real',
                                          line=dict(color=COLOR_ACTUAL, width=2)))
                if 'dentro_banda_95' in p.columns:
                    out = p[~p['dentro_banda_95'].astype(bool)]
                    if len(out):
                        figt.add_trace(go.Scatter(x=out['fecha'], y=out['cierre'], mode='markers',
                                                  name='Fuera de banda',
                                                  marker=dict(color='#FF4444', size=9, symbol='x')))
                figt.update_layout(plot_bgcolor='#808080', paper_bgcolor='#5A5A5A',
                                   font=dict(color='#FFF'), height=420, hovermode='x unified',
                                   margin=dict(l=60, r=20, t=20, b=50),
                                   yaxis=dict(tickformat='$,.0f', showgrid=False),
                                   xaxis=dict(showgrid=False))
                st.plotly_chart(figt, use_container_width=True)
                st.dataframe(hl.tail(15).iloc[::-1], use_container_width=True, hide_index=True)

else:
    st.markdown("""
    ### Qué hace esta herramienta

    **BTC Risk Bands** publica el rango probable del precio de Bitcoin, con la cobertura
    de ese rango medida y verificable. No pronostica dirección.

    - **Bandas de riesgo** — presupuesto para dimensionar posición y colocar stops
    - **Volatilidad esperada** — motor HAR-RV, el modelo estándar en pronóstico de volatilidad
    - **Regímenes** — quiebres estructurales y deriva del régimen vigente
    - **Contexto técnico** — medias, RSI, MACD y ATR, descriptivos, nunca como señal
    - **Track record auditable** — sellado con commits diarios en el repositorio

    **Lo que NO hace:** no promete ventaja direccional. Doce experimentos con protocolo
    anti-sesgo (holdout bloqueado, test anti-leakage con control positivo, walk-forward con
    costos, baselines obligatorios) no lograron demostrarla — y el producto se construyó
    sobre esa evidencia, no a pesar de ella.

    👈 Pulsa **Generar bandas** en el panel lateral.
    """)

st.markdown("---")
st.caption("© 2026 H&A DataLab Solutions SPA · Hinds-Analytics.com · "
           "Motor HAR-RV · Solo con fines informativos, no es asesoría financiera.")
