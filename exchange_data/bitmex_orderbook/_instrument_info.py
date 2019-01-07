import requests


class InstrumentInfo(object):
    INSTRUMENTS_URL = 'https://www.bitmex.com/api/v1/instrument?columns' \
                      '=symbol,tickSize&start=0&count=500'

    def __init__(
            self,
            index: int,
            symbol: str,
            tick_size: float,
            timestamp: str
    ):
        self.index = index
        self.timestamp = timestamp
        self.tick_size = tick_size
        self.symbol = symbol

    @staticmethod
    def get_instrument(symbol: str) -> 'InstrumentInfo':
        if not symbol.isupper():
            raise Exception('symbol should be uppercase.')

        r = requests.get(InstrumentInfo.INSTRUMENTS_URL)
        all_instruments = r.json()

        data = [data for data in all_instruments if data['symbol'] == symbol][0]

        index = all_instruments.index(data)

        return InstrumentInfo(
            index=index,
            symbol=symbol,
            tick_size=data['tickSize'],
            timestamp=data['timestamp']
        )