import mock

from exchange_data.channels import BitmexChannels
from exchange_data.trading._trade_executor import TradeExecutor


class TestTradeExecutor(object):
    @mock.patch('exchange_data.emitters.SignalInterceptor')
    @mock.patch('exchange_data.emitters.messenger.Messenger')
    def test_init(self, signal_mock, messenger_mock):
        executor = TradeExecutor('test', symbol='XBTUSD')

    @mock.patch('exchange_data.emitters.SignalInterceptor')
    @mock.patch('exchange_data.emitters.messenger.Messenger')
    def test_long_trade(self, signal_mock, messenger_mock):
        executor = TradeExecutor('test', symbol='XBTUSD')

        executor.long = mock.MagicMock()
        executor.execute(dict(data='Long'))
        executor.long.assert_called_once()

        executor.long = mock.MagicMock()
        executor.execute(dict(data='Long'))
        executor.long.assert_not_called()
