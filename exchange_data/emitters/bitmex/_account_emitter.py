import logging
import sys
from abc import ABC
from bitmex_websocket import Instrument
from bitmex_websocket._bitmex_websocket import BitMEXWebsocketConnectionError
from bitmex_websocket.auth import generate_nonce, generate_signature
from bitmex_websocket.constants import SecureInstrumentChannels, SecureChannels
from prometheus_client import Gauge, push_to_gateway, REGISTRY

from exchange_data import settings, Database, Measurement
from exchange_data.emitters import Messenger, SignalInterceptor

import alog
import click
import json
import signal
import websocket

from exchange_data.utils import DateTimeUtils

balance_guage = Gauge('account_balance', 'Account Balance', unit='BTC')

class BitmexAccountEmitter(
    Instrument,
    Messenger,
    Database,
    SignalInterceptor,
    DateTimeUtils,
    ABC
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
        Instrument.__init__(
            self,
            channels=self.channels,
            should_auth=True,
            **kwargs
        )
        Messenger.__init__(self)
        Database.__init__(self, database_name='bitmex')
        SignalInterceptor.__init__(self, self.stop)
        DateTimeUtils.__init__(self)

        self.on('action', self.on_action)

    def on_action(self, data):
        if isinstance(data, str):
            data = json.loads(data)

        table = data['table']
        channel_name = f'bitmex_{table}'

        self.log_data(data)
        m = self.measurement(table, data)

        self.write_points([m.__dict__], time_precision='ms')

        self.publish(channel_name, json.dumps(m.__dict__))

    def measurement(self, table, data):
        data_str = json.dumps(data)
        fields = dict(data=data_str)

        if table == 'wallet':
            amount = data['data'][0]['amount']
            amount = fields['amount'] = amount/(10**8)
            balance_guage.set(amount)
            self._push_metrics()

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

    def header(self):
        """Return auth headers. Will use API Keys if present in settings."""
        auth_header = []

        if self.should_auth:
            alog.info("Authenticating with API Key.")
            # To auth to the WS using an API key, we generate a signature
            # of a nonce and the WS API endpoint.
            alog.debug(settings.BITMEX_API_KEY)
            nonce = generate_nonce()
            api_signature = generate_signature(
                settings.BITMEX_API_SECRET, 'GET', '/realtime', nonce, '')

            auth_header = [
                "api-nonce: " + str(nonce),
                "api-signature: " + api_signature,
                "api-key:" + settings.BITMEX_API_KEY
            ]

        return auth_header

    def stop(self):
        sys.exit(0)

@click.command()
def main(**kwargs):
    emitter = BitmexAccountEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
