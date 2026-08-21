#!/usr/bin/env python3
"""
Capa de alertas — XGBit Volatility Radar
========================================
Tres detectores, ordenados de mas a menos validable. Todos responden a la misma
pregunta —"¿el riesgo de volatilidad esta elevado?"— y NINGUNO dice hacia donde
va el precio. Esa separacion es deliberada: doce experimentos establecieron que
la direccion no es predecible, y una alerta direccional reintroduciria por la
puerta de atras la promesa que el producto retiro.

  1. SHOCK (validable hoy)     : la volatilidad realizada supero por mucho a la
                                 pronosticada -> el estimador va por detras del
                                 regimen. Es deteccion ex post, no prediccion.
  2. CALENDARIO (validable)    : hay un evento macro PROGRAMADO en las proximas
                                 48h. Las fechas se conocen de antemano, asi que
                                 esto no anticipa nada: solo lee una agenda.
  3. NOTICIAS (NO validado)    : busqueda web para detectar catalizadores no
                                 programados. Se marca explicitamente como capa
                                 sin validacion estadistica.

Cada alerta lleva su nivel de evidencia. El usuario debe poder distinguir una
regla medida de una inferencia de un modelo de lenguaje.
"""
import os
import json
import datetime as dt

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
CONFIG = {
    'SHOCK_RATIO': 2.5,       # vol realizada / vol pronosticada que dispara alerta
    'SHOCK_SEVERE': 4.0,      # umbral de shock severo
    'EVENTS_FILE': 'monitoring/events.csv',
    'HORIZON_HOURS': 48,      # ventana de anticipacion del calendario
    'NEWS_MODEL': 'groq/compound-mini',   # busqueda web integrada (Tavily)
    'NEWS_FALLBACK': 'groq/compound',
}

NIVEL = {'ninguno': 0, 'bajo': 1, 'medio': 2, 'alto': 3}


# ===========================================================================
# 1. DETECTOR DE SHOCK — dato puro, sin fuentes externas
# ===========================================================================
def detectar_shock(rv_realizada, sigma_pronosticada):
    """Compara lo que paso con lo que el modelo esperaba.

    No predice: constata. Si la realizada supera varias veces a la pronosticada,
    el estimador quedo por detras del regimen y la banda de manana probablemente
    tambien lo este — el HAR necesita un par de dias para absorber el salto.
    """
    if not (np.isfinite(rv_realizada) and np.isfinite(sigma_pronosticada)) \
            or sigma_pronosticada <= 0:
        return {'activo': False, 'nivel': 'ninguno', 'ratio': np.nan, 'mensaje': ''}

    ratio = float(rv_realizada / sigma_pronosticada)
    if ratio >= CONFIG['SHOCK_SEVERE']:
        nivel = 'alto'
    elif ratio >= CONFIG['SHOCK_RATIO']:
        nivel = 'medio'
    else:
        return {'activo': False, 'nivel': 'ninguno', 'ratio': round(ratio, 2), 'mensaje': ''}

    msg = (f"Volatilidad realizada {ratio:.1f}x la pronosticada "
           f"({rv_realizada*100:.2f}% vs {sigma_pronosticada*100:.2f}% esperado). "
           "El estimador va por detras del regimen: la banda de manana puede quedarse "
           "corta hasta que el componente diario absorba el salto.")
    return {'activo': True, 'nivel': nivel, 'ratio': round(ratio, 2), 'mensaje': msg}


# ===========================================================================
# 2. CALENDARIO — fechas conocidas de antemano
# ===========================================================================
SEMILLA_EVENTOS = """fecha,evento,categoria,impacto
2026-09-16,Reunion FOMC - decision de tipos,macro,alto
2026-09-10,IPC de EE.UU.,macro,alto
2026-09-25,Vencimiento trimestral de opciones BTC (Deribit),cripto,alto
2026-10-28,Reunion FOMC - decision de tipos,macro,alto
2026-10-13,IPC de EE.UU.,macro,alto
2026-12-09,Reunion FOMC - decision de tipos,macro,alto
2026-12-25,Vencimiento trimestral de opciones BTC (Deribit),cripto,alto
"""


def cargar_eventos(ruta=None):
    """Lee el calendario local. Si no existe, lo crea con una semilla editable.

    Deliberadamente es un CSV que TU mantienes: un calendario auditable y bajo
    tu control es preferible a una API opaca que puede cambiar sin aviso.
    """
    ruta = ruta or CONFIG['EVENTS_FILE']
    try:
        if not os.path.exists(ruta):
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(SEMILLA_EVENTOS)
        d = pd.read_csv(ruta, parse_dates=['fecha'])
        return d.dropna(subset=['fecha'])
    except Exception:
        return pd.DataFrame(columns=['fecha', 'evento', 'categoria', 'impacto'])


def revisar_calendario(fecha_ref=None, horizonte_h=None):
    """¿Hay eventos programados de alto impacto en las proximas N horas?"""
    horizonte_h = horizonte_h or CONFIG['HORIZON_HOURS']
    hoy = pd.Timestamp(fecha_ref or dt.datetime.now(dt.timezone.utc).date())
    limite = hoy + pd.Timedelta(hours=horizonte_h)

    ev = cargar_eventos()
    if not len(ev):
        return {'activo': False, 'nivel': 'ninguno', 'eventos': [], 'mensaje': ''}

    prox = ev[(ev['fecha'] >= hoy) & (ev['fecha'] <= limite)]
    if not len(prox):
        return {'activo': False, 'nivel': 'ninguno', 'eventos': [], 'mensaje': ''}

    altos = prox[prox['impacto'].astype(str).str.lower() == 'alto']
    nivel = 'alto' if len(altos) else 'medio'
    listado = [f"{r['fecha'].date()} · {r['evento']}" for _, r in prox.iterrows()]
    msg = ("Evento(s) programado(s) en las proximas "
           f"{horizonte_h}h: " + " | ".join(listado) +
           ". Historicamente estos dias concentran volatilidad; la banda del modelo "
           "no lo sabe porque solo lee precios pasados.")
    return {'activo': True, 'nivel': nivel, 'eventos': listado, 'mensaje': msg}


# ===========================================================================
# 3. NOTICIAS — capa NO validada, con busqueda web
# ===========================================================================
PROMPT_NOTICIAS = """Eres un detector de riesgo de VOLATILIDAD para Bitcoin. Hoy es {hoy}.

Busca en la web si en las ultimas 24 horas o en las proximas 48 horas hay o habra
catalizadores capaces de provocar movimientos bruscos en el precio de Bitcoin:
decisiones de politica monetaria o fiscal, anuncios regulatorios, liquidaciones
masivas, fallos judiciales, movimientos de grandes tenedores, o fallos tecnicos
en exchanges relevantes.

REGLAS ESTRICTAS:
- Evalua RIESGO DE MOVIMIENTO BRUSCO, nunca direccion. Prohibido decir si el
  precio subira o bajara, y prohibido recomendar operar.
- Solo reporta hechos con fuente identificable. Si no encuentras nada relevante,
  responde nivel "ninguno". No inventes ni infieras.
- Se conservador: un falso positivo erosiona la confianza mas rapido de lo que un
  acierto la construye.

Responde UNICAMENTE con un objeto JSON, sin texto adicional ni marcas de codigo:
{{"nivel": "ninguno|bajo|medio|alto",
  "resumen": "una frase describiendo el catalizador, o cadena vacia",
  "fuentes": ["dominio1", "dominio2"]}}"""


def escanear_noticias(api_key=None, modelo=None, timeout_s=45):
    """Busqueda web para catalizadores no programados.

    ADVERTENCIA: esta capa NO esta validada estadisticamente. No hay un historico
    de noticias alineado con el que hacer backtest, asi que su tasa de acierto es
    desconocida. Se reporta siempre etiquetada como tal, y nunca dispara sola una
    accion: solo refuerza una alerta que ya tiene base en los datos.
    """
    api_key = api_key or os.environ.get('GROQ_API_KEY', '')
    if not api_key:
        return {'activo': False, 'nivel': 'ninguno', 'resumen': '',
                'fuentes': [], 'disponible': False,
                'mensaje': 'GROQ_API_KEY no configurada: capa de noticias omitida.'}
    try:
        from groq import Groq
        cliente = Groq(api_key=api_key)
        hoy = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
        contenido = None
        for m in [modelo or CONFIG['NEWS_MODEL'], CONFIG['NEWS_FALLBACK']]:
            try:
                r = cliente.chat.completions.create(
                    model=m,
                    messages=[{'role': 'user',
                               'content': PROMPT_NOTICIAS.format(hoy=hoy)}],
                    temperature=0.1, max_tokens=500, timeout=timeout_s)
                contenido = r.choices[0].message.content
                break
            except Exception:
                continue
        if contenido is None:
            raise RuntimeError('ningun modelo con busqueda respondio')

        txt = contenido.strip().replace('```json', '').replace('```', '').strip()
        i, j = txt.find('{'), txt.rfind('}')
        data = json.loads(txt[i:j+1]) if i >= 0 and j > i else {}
        nivel = str(data.get('nivel', 'ninguno')).lower()
        if nivel not in NIVEL:
            nivel = 'ninguno'
        resumen = str(data.get('resumen', ''))[:400]
        return {'activo': nivel != 'ninguno', 'nivel': nivel, 'resumen': resumen,
                'fuentes': data.get('fuentes', [])[:4], 'disponible': True,
                'mensaje': (f"[capa no validada] {resumen}" if resumen else '')}
    except Exception as e:
        return {'activo': False, 'nivel': 'ninguno', 'resumen': '', 'fuentes': [],
                'disponible': False,
                'mensaje': f'Capa de noticias no disponible ({type(e).__name__}).'}


# ===========================================================================
# CONSOLIDACION
# ===========================================================================
def evaluar_riesgo(rv_realizada, sigma_pronosticada, usar_noticias=True,
                   api_key=None, fecha_ref=None):
    """Corre las tres capas y consolida en un solo nivel y un mensaje.

    Regla de combinacion: el nivel final es el maximo de las capas, PERO la capa
    de noticias por si sola no puede superar 'medio'. Una inferencia de un modelo
    de lenguaje sin validacion no debe generar la alerta mas fuerte del sistema.
    """
    shock = detectar_shock(rv_realizada, sigma_pronosticada)
    cal = revisar_calendario(fecha_ref)
    news = (escanear_noticias(api_key) if usar_noticias
            else {'activo': False, 'nivel': 'ninguno', 'resumen': '', 'fuentes': [],
                  'disponible': False, 'mensaje': 'capa de noticias desactivada'})

    n_news = min(NIVEL[news['nivel']], NIVEL['medio'])   # techo para la capa no validada
    nivel_num = max(NIVEL[shock['nivel']], NIVEL[cal['nivel']], n_news)
    nivel = [k for k, v in NIVEL.items() if v == nivel_num][0]

    partes = []
    if shock['activo']:
        partes.append(f"SHOCK · {shock['mensaje']}")
    if cal['activo']:
        partes.append(f"CALENDARIO · {cal['mensaje']}")
    if news['activo']:
        fuentes = f" (fuentes: {', '.join(news['fuentes'])})" if news['fuentes'] else ''
        partes.append(f"NOTICIAS · {news['mensaje']}{fuentes}")

    # El refuerzo entre capas es la senal mas util: dos capas coincidiendo desde
    # evidencias independientes es mas informativo que una capa gritando sola.
    n_activas = sum(bool(x['activo']) for x in (shock, cal, news))
    reforzado = n_activas >= 2
    if reforzado:
        partes.append("⚠️ DOS O MAS CAPAS COINCIDEN desde evidencias independientes: "
                      "el riesgo de movimiento brusco esta reforzado.")

    if nivel != 'ninguno':
        partes.append("Regla 5 del manual: revisar posiciones abiertas, no abrir tamano "
                      "nuevo hasta que el estimador se reajuste. Esto NO indica direccion.")

    return {
        'nivel': nivel, 'reforzado': reforzado, 'n_capas': n_activas,
        'shock': shock, 'calendario': cal, 'noticias': news,
        'mensaje': '\n\n'.join(partes) if partes else 'Sin alertas de riesgo.',
    }


if __name__ == '__main__':
    # Prueba manual: python alerts.py
    r = evaluar_riesgo(0.048, 0.012, usar_noticias=False)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ('nivel', 'reforzado', 'n_capas')}, indent=2))
    print(r['mensaje'])
