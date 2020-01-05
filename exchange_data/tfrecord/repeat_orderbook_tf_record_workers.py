from pathlib import Path

from exchange_data.emitters import Messenger
from exchange_data.tfrecord.orderbook_tf_record_workers import OrderBookTFRecordWorkers
from exchange_data.utils import DateTimeUtils


class RepeatOrderBoekTFRecordWorkers(Messenger):
    def __init__(self, repeat_interval, directory_name, **kwargs):
        self.directory = Path(f'{Path.home()}/.exchange-data/tfrecords/{directory_name}')
        self.repeat_interval = repeat_interval
        kwargs['directory_name'] = directory_name
        kwargs['directory'] = self.directory
        self.kwargs = kwargs

        super().__init__()

        self.on(repeat_interval, self.run_workers)

    def run_workers(self, timestamp):
        start_date = DateTimeUtils.now()
        OrderBookTFRecordWorkers(**self.kwargs).run()
        self.publish('OrderBookTFRecordWorkers', str(start_date))

    def run(self):
        if self.repeat_interval:
            self.sub([self.repeat_interval, 'resnet_trainer_done'])
        else:
            self.run_workers(None)
