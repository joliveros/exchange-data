from . import Recorder
from gdax import OrderBook
import alog


class GdaxRecorder(Recorder, OrderBook):
    measurements = []

    def __init__(self, symbols: list):
        Recorder.__init__(self, symbols=symbols, database_name='gdax')
        OrderBook.__init__(self, product_id=symbols)

        self.start()

    def on_message(self, msg):
        alog.debug(msg)
        self.save_measurement('level2', msg['product_id'], msg)
