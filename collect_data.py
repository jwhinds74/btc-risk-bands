#!/usr/bin/env python3
"""Colector diario XGBit: actualiza todos los caches de datos. Disenado para GitHub Actions."""
import os, sys, importlib.util
# Reutiliza las funciones de fetch del notebook exportadas aqui de forma autonoma:
import requests, numpy as np, pandas as pd, datetime as dt

CACHES = {'price':'btc_ohlcv_cache.csv','funding':'funding_cache.csv','oi':'oi_cache.csv',
          'eth':'eth_cache.csv','ndx':'macro_cache_ndx.csv','dxy':'macro_cache_dxy.csv',
          'chain':'chain_cache.csv','mvrv':'mvrv_cache.csv'}

def merge_save(new, path):
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=[0], index_col=0)
        new = pd.concat([old, new]).groupby(level=0).last().sort_index()
    new.to_csv(path); print(f'{path}: {len(new)} filas')

def main():
    ok, fail = [], []
    def step(name, fn):
        try: fn(); ok.append(name)
        except Exception as e: fail.append(f'{name}({type(e).__name__})')
    key = os.environ.get('CRYPTOCOMPARE_API_KEY','')
    def binance(sym, cols, cache):
        # Fallback sin API key. CryptoCompare devuelve 401 con la key revocada,
        # asi que el precio y ETH deben poder venir de aqui o el colector nace
        # con 2 fallos fijos (y su tolerancia es <= 2).
        rows, end = [], None
        for _ in range(2):
            p = {'symbol': f'{sym}USDT', 'interval': '1d', 'limit': 500}
            if end: p['endTime'] = end
            r = requests.get('https://api.binance.com/api/v3/klines', params=p, timeout=20)
            r.raise_for_status(); k = r.json()
            if not k: break
            rows = k + rows; end = k[0][0] - 1
            if len(k) < 500: break
        d = pd.DataFrame(rows, columns=['t','open','high','low','close','vol',
                                        'ct','volumeto','n','tb','tq','ig'])
        d['ts'] = pd.to_datetime(d['t'], unit='ms')
        d = d.set_index('ts')[cols].astype(float)
        merge_save(d[~d.index.duplicated(keep='last')].sort_index(), cache)

    def cc(sym, cols, cache):
        try:
            p = {'fsym':sym,'tsym':'USD','limit':400}
            if key: p['api_key'] = key
            r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday', params=p, timeout=20)
            r.raise_for_status(); d = pd.DataFrame(r.json()['Data']['Data'])
            if d.empty: raise RuntimeError('respuesta vacia')
            d['ts'] = pd.to_datetime(d['time'], unit='s')
            merge_save(d.set_index('ts')[cols], cache)
        except Exception as e:
            print(f'  CryptoCompare {sym} fallo ({type(e).__name__}) -> Binance')
            binance(sym, cols, cache)
    step('price', lambda: cc('BTC', ['close','high','low','volumeto'], CACHES['price']))
    step('eth',   lambda: cc('ETH', ['close'], CACHES['eth']))
    def funding():
        r = requests.get('https://fapi.binance.com/fapi/v1/fundingRate',
                         params={'symbol':'BTCUSDT','limit':1000}, timeout=20)
        r.raise_for_status(); f = pd.DataFrame(r.json())
        f['ts'] = pd.to_datetime(f['fundingTime'], unit='ms'); f['fundingRate']=f['fundingRate'].astype(float)
        g = f.set_index('ts')['fundingRate']
        out = pd.concat([g.resample('1D').mean().rename('funding'),
                         g.resample('1D').apply(lambda s: s.iloc[s.abs().argmax()] if len(s) else np.nan).rename('funding_absmax')], axis=1)
        merge_save(out, CACHES['funding'])
    step('funding', funding)
    def oi():
        r = requests.get('https://fapi.binance.com/futures/data/openInterestHist',
                         params={'symbol':'BTCUSDT','period':'1d','limit':500}, timeout=20)
        r.raise_for_status(); o = pd.DataFrame(r.json())
        o['ts'] = pd.to_datetime(o['timestamp'], unit='ms').dt.normalize()
        merge_save(o.set_index('ts')[['sumOpenInterest']].astype(float).rename(columns={'sumOpenInterest':'oi'}), CACHES['oi'])
    step('oi', oi)
    def stooq(sym, col, cache):
        from io import StringIO
        r = requests.get(f'https://stooq.com/q/d/l/?s={sym}&i=d', timeout=20); r.raise_for_status()
        d = pd.read_csv(StringIO(r.text), parse_dates=['Date'])
        merge_save(d.set_index('Date')[['Close']].rename(columns={'Close':col}), cache)
    step('ndx', lambda: stooq('%5Endx','ndx',CACHES['ndx']))
    step('dxy', lambda: stooq('dx.f','dxy',CACHES['dxy']))
    def chain():
        def one(ch):
            r = requests.get(f'https://api.blockchain.info/charts/{ch}',
                             params={'timespan':'180days','format':'json'}, timeout=25)
            r.raise_for_status(); v = pd.DataFrame(r.json()['values'])
            v['ts'] = pd.to_datetime(v['x'], unit='s').dt.normalize()
            return v.set_index('ts')['y'].groupby(level=0).mean()
        merge_save(pd.DataFrame({'active_addr':one('n-unique-addresses'),'n_tx':one('n-transactions'),
                                 'onchain_vol':one('estimated-transaction-volume-usd'),
                                 'trade_vol':one('trade-volume'),'hash_rate':one('hash-rate')}), CACHES['chain'])
    step('chain', chain)
    def mvrv():
        r = requests.get('https://community-api.coinmetrics.io/v4/timeseries/asset-metrics',
                         params={'assets':'btc','metrics':'CapMVRVCur','frequency':'1d','page_size':500}, timeout=25)
        r.raise_for_status(); d = pd.DataFrame(r.json()['data'])
        d['ts'] = pd.to_datetime(d['time']).dt.tz_localize(None).dt.normalize()
        d['mvrv'] = d['CapMVRVCur'].astype(float)
        merge_save(d.set_index('ts')[['mvrv']], CACHES['mvrv'])
    step('mvrv', mvrv)
    print(f'OK: {ok} | FALLOS: {fail}')
    sys.exit(0 if len(fail) <= 2 else 1)   # tolerante a caidas puntuales de fuentes

if __name__ == '__main__':
    main()
