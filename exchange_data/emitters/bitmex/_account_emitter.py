#!/usr/bin/env python
import os
from abc import ABC
from bitmex_websocket import Instrument
from bitmex_websocket._bitmex_websocket import BitMEXWebsocketConnectionError
from bitmex_websocket.auth import generate_nonce, generate_signature
from bitmex_websocket.constants import SecureInstrumentChannels, SecureChannels
from exchange_data import settings, Database, Measurement
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.utils import DateTimeUtils
from prometheus_client import Gauge, push_to_gateway, REGISTRY

import alog
import click
import json
import logging
import signal
import sys
import websocket

balance_guage = Gauge('account_balance', 'Account Balance', unit='BTC')


class BitmexAccountEmitter(
    Database,
    Messenger,
    Instrument,
    DateTimeUtils,
    # SignalInterceptor,
):
    measurements = []
    channels = [
        SecureChannels.margin,
        SecureChannels.transact,
        SecureChannels.wallet,
        SecureInstrumentChannels.execution,
        SecureInstrumentChannels.order,
        SecureInstrumentChannels.position
    ]

    secure_channels = [
        SecureChannels.margin,
        SecureChannels.transact,
        SecureChannels.wallet
    ]

    def __init__(self, **kwargs):
        websocket.enableTrace(settings.RUN_ENV == 'development')
        BITMEX_API_KEY = os.environ.get('BITMEX_API_KEY')
        BITMEX_API_SECRET = os.environ.get('BITMEX_API_SECRET')

        alog.info((BITMEX_API_KEY, BITMEX_API_SECRET))

        super().__init__(
            channels=self.channels,
            should_auth=True,
            database_name='bitmex',
            # exit_func=self.stop,
            **kwargs
        )

        self.on('action', self.on_action)

    def on_action(self, data):
        # alog.info(alog.pformat(data))

        if isinstance(data, str):
            data = json.loads(data)

        table = data['table']
        channel_name = f'bitmex_{table}'

        m = self.measurement(table, data)

        alog.info(m)

        self.write_points([m.__dict__], time_precision='ms')

        self.publish(channel_name, str(m))

    def measurement(self, table, data):
        data_str = json.dumps(data)
        fields = dict(data=data_str)

        alog.info(table)

        # if table == 'wallet':
        #     amount = data['data'][0]['amount']
        #     amount = fields['amount'] = amount/(10**8)
        #     balance_guage.set(amount)
        #     self._push_metrics()

        return Measurement(
            measurement=table,
            time=self.now(),
            tags={},
            fields=fields
        )

    def log_data(self, data):
        if settings.LOG_LEVEL == logging.DEBUG:
            if data['table'] == 'wallet':
                alog.info(alog.pformat(data))

    def _push_metrics(self):
        push_to_gateway(
            settings.PROMETHEUS_HOST,
            job='account',
            registry=REGISTRY
        )

    def start(self):
        self.run_forever()

    def on_error(self, error=None, msg=None):
        raise BitMEXWebsocketConnectionError(error)

    def subscribe_channels(self):
        for channel in self.channels:

            if channel in self.secure_channels:
                channel_key = channel.name
            else:
                channel_key = f'{channel.name}:{self.symbol}'
            self.subscribe(channel_key)

    def stop(self):
        sys.exit(0)


@click.command()
def main(**kwargs):
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # alog.info(alog.pformat(loggers))
    # logging.getLogger('requests').setLevel(logging.DEBUG)
    emitter = BitmexAccountEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
