from exchange_data.emitters import Messenger
from exchange_data.utils import NoValue
from mock import MagicMock

import alog
import pytest


class TestMessenger(object):

    def test_messenger_subscribe_should_fail_if_channel_is_string(self, mocker):
        mocker.patch('exchange_data.emitters.messenger.Redis')

        messenger = Messenger()
        messenger.pubsub = MagicMock()

        with pytest.raises(Exception):
            messenger.sub('tick')

    def test_messenger_subscribe_should_only_accept_enum(self, mocker):

        class Channels(NoValue):
            Tick = 'tick'

        mocker.patch('exchange_data.emitters.messenger.Redis')

        messenger = Messenger()
        messenger.pubsub = MagicMock()

        messenger.sub([Channels.Tick])
