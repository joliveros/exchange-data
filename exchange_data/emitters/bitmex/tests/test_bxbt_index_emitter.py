from exchange_data.emitters.bitmex._bxbt_index_emitter import BXBTIndexEmitter

import dateutil
import mock
import pytest


class TestBXBTIndexEmitter(object):

    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.Messenger')
    @pytest.mark.vcr()
    def test_init(self, messenger_mock):
        emitter = BXBTIndexEmitter(interval='1m')

        assert emitter.channel == '.BXBT_1m'

    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.Messenger')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.BXBTIndexEmitter.publish')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.BXBTIndexEmitter.setup_signals')
    @mock.patch('exchange_data.emitters.bitmex._bxbt_index_emitter.Database.write_points')
    @pytest.mark.vcr()
    def test_fetch_price(self, messenger_mock, publish_mock, setup_signals_mock, write_points_mock):
        emitter = BXBTIndexEmitter(interval='1m', influxdb='http://jose:jade121415@0.0.0.0:28953/')
        start_time = dateutil.parser.parse('2019-02-22 21:35:42.784633')
        emitter.fetch_price('', start_time)

