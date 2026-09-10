import base64
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import requests

from prop_trader.upbit_broker import (make_token,order_payload,submit,BrokerError,UpbitClient,OrderJournal)


def client():
    fake=Mock()
    chance=dict(market=dict(state='active',bid_types=['price'],ask_types=['market'],
        bid={'min_total':'5000'},ask={'min_total':'5000'},max_total='100000000'),
        bid_fee='0.0005',bid_account={'balance':'100000'},ask_account={'balance':'1'})
    def call(method,path,params=None,private=True):
        if path=='/v1/orders/chance':return chance
        if path=='/v1/ticker':return [{'trade_price':100000000}]
        return dict(uuid='mock-order',state='wait',executed_volume='0')
    fake.request.side_effect=call
    return fake


class BrokerTests(unittest.TestCase):
    def test_auth_signature_and_query_hash(self):
        token=make_token('example-access','example-secret',{'market':'KRW-BTC','price':'10000'},nonce='fixed')
        head,payload,sig=token.split('.')
        decoded=json.loads(base64.urlsafe_b64decode(payload+'='*(-len(payload)%4)))
        self.assertEqual(decoded['query_hash'],hashlib.sha512(b'market=KRW-BTC&price=10000').hexdigest())
        expected=hmac.new(b'example-secret',(head+'.'+payload).encode(),hashlib.sha512).digest()
        self.assertEqual(base64.urlsafe_b64decode(sig+'='*(-len(sig)%4)),expected)

    def test_paper_no_network_and_units(self):
        fake=client();p=order_payload('KRW-BTC','buy','10000','paper1')
        self.assertNotIn('volume',p)
        self.assertFalse(submit(fake,p)['submitted']);fake.request.assert_not_called()
        sell=order_payload('KRW-BTC','sell','.0001','paper2')
        self.assertNotIn('price',sell)
        with self.assertRaises(BrokerError):order_payload('BTCUSDT','buy',10000,'x')
        with self.assertRaises(BrokerError):order_payload('KRW-BTC','buy','NaN','x')

    def test_exchange_test_never_uses_live_endpoint(self):
        fake=client();submit(fake,order_payload('KRW-BTC','buy',10000,'test1'),mode='test')
        self.assertEqual(fake.request.call_args.args[1],'/v1/orders/test')

    def test_live_duplicate_submission_blocked(self):
        fake=client();p=order_payload('KRW-BTC','buy',10000,'live1')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'orders.db'
            r=submit(fake,p,'live',path)
            self.assertEqual(r['response']['state'],'wait') # accepted != filled
            with self.assertRaises(BrokerError):submit(fake,p,'live',path)
            self.assertEqual(sum(c.args[:2]==('POST','/v1/orders') for c in fake.request.call_args_list),1)

    def test_unknown_submission_is_not_retried(self):
        fake=client();normal=fake.request.side_effect
        def error(method,path,*args,**kwargs):
            if method=='POST':raise BrokerError('timeout')
            return normal(method,path,*args,**kwargs)
        fake.request.side_effect=error;p=order_payload('KRW-BTC','buy',10000,'unknown1')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'orders.db'
            with self.assertRaises(BrokerError):submit(fake,p,'live',path)
            journal=OrderJournal(path)
            self.assertEqual(journal.db.execute('SELECT status FROM orders').fetchone()[0],'UNKNOWN')
            with self.assertRaises(BrokerError):submit(fake,p,'live',path)
            self.assertEqual(sum(c.args[0]=='POST' for c in fake.request.call_args_list),1)

    def test_unheld_sell_and_cap_block_orders(self):
        fake=client()
        with self.assertRaises(BrokerError):submit(fake,order_payload('KRW-BTC','sell',2,'sell1'),'test')
        with self.assertRaises(BrokerError):submit(fake,order_payload('KRW-BTC','buy',20000,'buy1'),'test')
        self.assertFalse(any(c.args[0]=='POST' for c in fake.request.call_args_list))

    def test_transport_timeout_and_no_secret_in_error(self):
        session=Mock();session.request.side_effect=requests.Timeout('secret-value')
        c=UpbitClient('access','secret-value',session)
        with self.assertRaises(BrokerError) as result:c.request('GET','/v1/accounts')
        self.assertNotIn('secret-value',str(result.exception))
        self.assertEqual(session.request.call_count,1)
