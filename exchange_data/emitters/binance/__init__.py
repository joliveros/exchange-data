from datetime import datetime
from binance.exceptions import BinanceAPIException
from cached_property import cached_property_with_ttl, cached_property
from dateutil.tz import tz

from exchange_data.emitters.binance.proxied_client import ProxiedClient
from exchange_data.utils import DateTimeUtils

import alog
import re
import time


class BinanceUtils(object):
    @cached_property
    def client(self):
        return ProxiedClient()

    @cached_property_with_ttl(ttl=60 * 10)
    def symbols(self):
        return self.get_symbols()

    def get_symbols(self):
        try:
            return self._get_symbols()
        except BinanceAPIException as e:
            alog.info(e.message)

            self.sleep_during_embargo(e)

            return []

    def _get_symbols(self):
        exchange_info = self.client.get_exchange_info()

        symbols = [symbol for symbol in exchange_info['symbols']
                   if symbol['status'] == 'TRADING']

        symbol_names = [symbol['symbol'] for symbol in symbols if symbol[
            'symbol']]

        return symbol_names

    def sleep_during_embargo(self, e):
        if e.status_code == 418:
            embargo_timestamp = int(re.search('\d+', e.message)[0])
            embargo_timestamp = datetime.fromtimestamp(embargo_timestamp / 1000) \
                .replace(tzinfo=tz.tzlocal())
            alog.info(f'banned until {embargo_timestamp}')
            diff = embargo_timestamp - DateTimeUtils.now()

            sleep_seconds = diff.total_seconds()

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
