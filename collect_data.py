#!/usr/bin/env python3
"""
Colector diario — BTC Risk Bands (motor HAR-RV)
===============================================
Baja UNICAMENTE el OHLCV diario de BTC. Nada mas.

Por que tan corto: el colector heredado de la version XGBoost bajaba ademas ETH,
funding, open interest, NDX, DXY, on-chain y MVRV. Todos eran insumos de
experimentos cross-asset ya falsificados, y HAR-RV no usa ninguno: solo necesita
cierre, maximo y minimo.

Ademas, en los runners de GitHub Actions varias de esas fuentes fallan de forma
sistematica (Binance bloquea IPs de datacenter estadounidenses; Yahoo devuelve
429 ante trafico de datacenter), lo que dejaba el workflow en rojo permanente.
Un rojo cronico entrena a ignorar la senal, y entonces el dia que se rompe algo
de verdad nadie se entera. Este colector sale verde cuando consigue los datos y
rojo cuando no — la unica forma de que el indicador signifique algo.

Jerarquia: CryptoCompare (si hay key) -> Binance klines -> conserva el cache.
"""
import os
import sys
import datetime as dt

import pandas as pd
import requests

CACHE = 'btc_ohlcv_cache.csv'
COLS = ['close', 'high', 'low']


def log(m):
    print(f'[collect] {m}', flush=True)


def desde_cryptocompare():
    key = os.environ.get('CRYPTOCOMPARE_API_KEY', '')
    headers = {'authorization': f'Apikey {key}'} if key else {}
    r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday',
                     params={'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000},
                     headers=headers, timeout=25)
    r.raise_for_status()
    j = r.json()
    if j.get('Response') != 'Success':
        raise RuntimeError(j.get('Message', 'respuesta inesperada'))
    d = pd.DataFrame(j['Data']['Data'])
    if d.empty:
        raise RuntimeError('respuesta vacia')
    d['ts'] = pd.to_datetime(d['time'], unit='s')
    return d.set_index('ts')[COLS].astype(float)


def desde_binance():
    rows, end = [], None
    for _ in range(4):
        p = {'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 1000}
        if end:
            p['endTime'] = end
        r = requests.get('https://api.binance.com/api/v3/klines', params=p, timeout=25)
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        rows = k + rows
        end = k[0][0] - 1
        if len(k) < 1000:
            break
    if not rows:
        raise RuntimeError('sin velas')
    d = pd.DataFrame(rows, columns=['t', 'o', 'high', 'low', 'close', 'v',
                                    'ct', 'q', 'n', 'tb', 'tq', 'ig'])
    d['ts'] = pd.to_datetime(d['t'], unit='ms')
    d = d.set_index('ts')[COLS].astype(float)
    return d[~d.index.duplicated(keep='last')].sort_index()


def main():
    nuevo, fuente = None, None
    for nombre, fn in (('CryptoCompare', desde_cryptocompare),
                       ('Binance', desde_binance)):
        try:
            nuevo = fn()
            fuente = nombre
            log(f'{nombre}: {len(nuevo)} velas')
            break
        except Exception as e:
            log(f'{nombre} fallo ({type(e).__name__}: {e})')

    if nuevo is None:
        if os.path.exists(CACHE):
            log('ADVERTENCIA: ninguna fuente respondio; se conserva el cache. '
                'monitor.py puede seguir operando con el historial existente.')
        else:
            log('ERROR: ninguna fuente respondio y no hay cache.')
        sys.exit(1)

    # Fusion con el cache: conserva la historia previa y anade las velas nuevas.
    if os.path.exists(CACHE):
        viejo = pd.read_csv(CACHE, parse_dates=[0], index_col=0)
        for c in COLS:
            if c not in viejo.columns:
                viejo[c] = viejo['close']
        combinado = pd.concat([viejo[COLS], nuevo])
        combinado = combinado[~combinado.index.duplicated(keep='last')].sort_index()
        agregadas = len(combinado) - len(viejo)
    else:
        combinado, agregadas = nuevo, len(nuevo)

    # Solo velas cerradas: nada con fecha de hoy UTC.
    hoy = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
    combinado = combinado[combinado.index < hoy]

    if len(combinado) < 300:
        log(f'ERROR: solo {len(combinado)} velas; HAR-RV necesita 300+.')
        sys.exit(1)

    combinado.index.name = 'ts'
    combinado.to_csv(CACHE)
    log(f'{CACHE}: {len(combinado)} filas (+{max(agregadas, 0)} nuevas) via {fuente}')
    log(f'rango: {combinado.index[0].date()} -> {combinado.index[-1].date()}')
    sys.exit(0)


if __name__ == '__main__':
    main()
