"""KRW spot execution, default paper; no credentials are written to artifacts."""
import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode, unquote
import uuid
import time
import threading

import requests


class BrokerError(RuntimeError):
    pass


def decimal(value):
    try:
        result=Decimal(str(value))
    except InvalidOperation:
        raise BrokerError('Invalid numeric amount') from None
    if not result.is_finite() or result<=0:
        raise BrokerError('Amount must be finite and positive')
    return result


def market_code(market):
    if not re.fullmatch(r'KRW-[A-Z0-9]+',market):
        raise BrokerError('Only explicit Upbit KRW spot markets are supported (e.g. KRW-BTC)')
    return market


def make_token(access,secret,params=None,nonce=None):
    def encode(obj):
        return base64.urlsafe_b64encode(json.dumps(obj,separators=(',',':')).encode()).rstrip(b'=')
    payload=dict(access_key=access,nonce=nonce or str(uuid.uuid4()))
    if params:
        query=unquote(urlencode(params,doseq=True))
        payload.update(query_hash=hashlib.sha512(query.encode()).hexdigest(),query_hash_alg='SHA512')
    body=encode(dict(alg='HS512',typ='JWT'))+b'.'+encode(payload)
    signature=base64.urlsafe_b64encode(hmac.new(secret.encode(),body,hashlib.sha512).digest()).rstrip(b'=')
    return (body+b'.'+signature).decode()


class UpbitClient:
    _rate_lock = threading.Lock()
    _next_request = 0.
    _blocked_until = 0.
    def __init__(self,access=None,secret=None,session=None):
        self._access=access if access is not None else os.environ.get('UPBIT_ACCESS_KEY')
        self._secret=secret if secret is not None else os.environ.get('UPBIT_SECRET_KEY')
        self.session=session or requests.Session()

    def request(self,method,path,params=None,private=True):
        if private and not (self._access and self._secret):
            raise BrokerError('Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY locally before authenticated requests')
        with self._rate_lock:
            now = time.monotonic()
            if now < UpbitClient._blocked_until:
                raise BrokerError('API 요청 제한 대기 중; 주문 결과 확인 후 재개')
            time.sleep(max(0., UpbitClient._next_request-now))
            UpbitClient._next_request = time.monotonic()+.15
        headers={'Accept':'application/json'}
        if private:headers['Authorization']='Bearer '+make_token(self._access,self._secret,params)
        kwargs=dict(headers=headers,timeout=(5,15),allow_redirects=False)
        if method=='POST':kwargs['json']=params
        else:kwargs['params']=params
        try:
            response=self.session.request(method,'https://api.upbit.com'+path,**kwargs)
        except requests.RequestException:
            raise BrokerError('Transport failure; submission outcome may be unknown. Reconcile identifier; do not resubmit.') from None
        if not 200<=response.status_code<300:
            if response.status_code in (429, 418):
                with self._rate_lock:
                    UpbitClient._blocked_until = time.monotonic() + (60 if response.status_code==429 else 300)
            raise BrokerError(f'Upbit HTTP {response.status_code}; no automatic retry')
        try:return response.json()
        except ValueError:raise BrokerError('Malformed response; reconcile any submitted order') from None


class OrderJournal:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS orders (identifier TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,response TEXT)')
        self.db.commit()

    def claim(self,identifier,payload):
        encoded=json.dumps(payload,sort_keys=True)
        with self.db:
            row=self.db.execute('SELECT payload,status FROM orders WHERE identifier=?',(identifier,)).fetchone()
            if row:
                if row[0]!=encoded:raise BrokerError('Identifier already used with a different order')
                raise BrokerError(f'Identifier already recorded ({row[1]}); query status instead of resubmitting')
            self.db.execute('INSERT INTO orders VALUES (?,?,?,?)',(identifier,encoded,'SUBMITTING',None))

    def update(self,identifier,status,response):
        with self.db:
            self.db.execute('UPDATE orders SET status=?,response=? WHERE identifier=?',(status,json.dumps(response),identifier))


def order_payload(market,side,amount,identifier):
    market_code(market)
    if side not in ('buy','sell'):raise BrokerError('side must be buy or sell')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',identifier):raise BrokerError('identifier must be 1..40 alphanumeric, dash or underscore characters')
    value=format(decimal(amount),'f')
    payload=dict(market=market,side='bid' if side=='buy' else 'ask',
                 ord_type='price' if side=='buy' else 'market',identifier=identifier)
    payload['price' if side=='buy' else 'volume']=value
    return payload


def validate_account(client,payload,max_krw,owned_volume=None):
    chance=client.request('GET','/v1/orders/chance',dict(market=payload['market']))
    market=chance.get('market',{})
    if market.get('state')!='active':raise BrokerError('Market is not active')
    side=payload['side'];order_type=payload['ord_type']
    if order_type not in market.get(side+'_types',chance.get(side+'_types',[])):
        raise BrokerError('Market does not support requested order type')
    if side=='bid':
        total=decimal(payload['price'])
        fee=Decimal(str(chance['bid_fee']))
        if not fee.is_finite() or fee<0:raise BrokerError('Invalid exchange fee')
        if Decimal(str(chance['bid_account']['balance']))<total*(1+fee):
            raise BrokerError('Insufficient available KRW including fee')
    else:
        quantity=decimal(payload['volume'])
        if Decimal(str(chance['ask_account']['balance']))<quantity:
            raise BrokerError('Insufficient available asset; spot selling cannot create a short')
        ticker=client.request('GET','/v1/ticker',dict(markets=payload['market']),private=False)
        total=quantity*decimal(ticker[0]['trade_price'])
    minimum=decimal(market[side]['min_total'])
    if total<minimum:raise BrokerError('Order below exchange minimum notional')
    if side=='ask' and owned_volume is not None:
        if quantity>decimal(owned_volume):raise BrokerError('Sell exceeds bot-owned quantity')
    elif total>decimal(max_krw):raise BrokerError('Order exceeds configured KRW notional cap')
    if market.get('max_total') and total>decimal(market['max_total']):
        raise BrokerError('Order exceeds exchange maximum')
    return dict(estimated_notional_krw=str(total),minimum_krw=str(minimum))


def submit(client,payload,mode='paper',journal_path=Path('runs/upbit/orders.sqlite3'),max_krw='10000',owned_volume=None,deadline=None):
    if mode not in ('paper','test','live'):raise BrokerError('Unknown execution mode')
    # Revalidate caller-provided payload, including market units, before any network access.
    expected=order_payload(payload['market'],'buy' if payload.get('side')=='bid' else 'sell',
                           payload.get('price',payload.get('volume')),payload['identifier'])
    if payload!=expected:raise BrokerError('Unsupported or malformed order payload')
    if mode=='paper':
        if payload['side']=='bid' and decimal(payload['price'])>decimal(max_krw):
            raise BrokerError('Order exceeds configured KRW cap')
        return dict(mode='paper',submitted=False,payload=payload,note='Preview only; balances, minimum and market availability not checked')
    policy=validate_account(client,payload,max_krw,owned_volume)
    if deadline is not None and time.time() >= deadline:
        raise BrokerError('호가 또는 다중 주기 예측 만료: 주문 보류')
    if mode=='test':
        response=client.request('POST','/v1/orders/test',payload)
        return dict(mode='test',submitted=False,policy=policy,response=response)
    journal=OrderJournal(journal_path)
    try:
        journal.claim(payload['identifier'],payload)
        try:
            response=client.request('POST','/v1/orders',payload)
            if not isinstance(response,dict) or not response.get('uuid'):
                raise BrokerError('Missing order UUID; reconcile identifier')
        except Exception:
            journal.update(payload['identifier'],'UNKNOWN',{'action':'reconcile_identifier'})
            raise
        journal.update(payload['identifier'],'SUBMITTED',response)
        return dict(mode='live',submitted=True,policy=policy,response=response)
    finally:
        journal.db.close()



def main():
    p=argparse.ArgumentParser(description='Upbit KRW spot execution; defaults to paper preview')
    p.add_argument('action',choices=['buy','sell','status','cancel','balances'])
    p.add_argument('--market');p.add_argument('--krw');p.add_argument('--volume');p.add_argument('--identifier')
    p.add_argument('--mode',choices=['paper','test','live'],default='paper')
    p.add_argument('--max-krw',default=os.environ.get('UPBIT_MAX_ORDER_KRW','10000'))
    args=p.parse_args();client=UpbitClient()
    if args.action in ('buy','sell'):
        if not args.market or not args.identifier:p.error('--market and persistent --identifier are required')
        if args.action=='buy' and (not args.krw or args.volume):p.error('buy requires --krw only')
        if args.action=='sell' and (not args.volume or args.krw):p.error('sell requires --volume only')
        result=submit(client,order_payload(args.market,args.action,args.krw or args.volume,args.identifier),args.mode,max_krw=args.max_krw)
    elif args.action=='balances':result=client.request('GET','/v1/accounts')
    else:
        if not args.identifier:p.error('--identifier required')
        if args.action=='cancel' and args.mode!='live':
            result=dict(mode=args.mode,submitted=False,action='cancel',identifier=args.identifier)
        else:
            result=client.request('DELETE' if args.action=='cancel' else 'GET','/v1/order',dict(identifier=args.identifier))
            OrderJournal(Path('runs/upbit/orders.sqlite3')).update(args.identifier,result.get('state','UNKNOWN'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
