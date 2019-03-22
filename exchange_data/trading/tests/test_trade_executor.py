import mock

from exchange_data.trading._trade_executor import TradeExecutor


class TestTradeExecutor(object):

    @mock.patch('exchange_data.emitters.SignalInterceptor')
    @mock.patch('exchange_data.emitters.messenger.Messenger')
    def test_long_trade(self, signal_mock, messenger_mock):
        exec=TradeExecutor('test')

        exec.execute(dict(data='Long'))
