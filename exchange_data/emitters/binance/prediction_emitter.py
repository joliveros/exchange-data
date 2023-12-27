#!/usr/bin/env python
from exchange_data import Database, Measurement
from exchange_data.data.orderbook_dataset import orderbook_dataset
from exchange_data.emitters import Messenger
from exchange_data.emitters.binance import BinanceUtils
from os.path import realpath
from transformers import ViTFeatureExtractor, ViTForImageClassification
import alog
import click

from exchange_data.utils import DateTimeUtils


device = "cuda:0"
PATH = realpath("./vit_output/pretrained")
# model = model.to(device)


class PredictionEmitter(Messenger, Database, BinanceUtils):
    def __init__(self, tick, tick_interval, **kwargs):
        kwargs['futures'] = True
        self.symbol = kwargs['symbol']
        self._kwargs = kwargs

        super().__init__(**kwargs)
        Database.__init__(self, **kwargs)
        BinanceUtils.__init__(self, **kwargs)

        self.prediction_channel = f"{self.symbol}_prediction"

        if tick:
            self.tick()
        else:
            self.on(tick_interval, self.tick)

            self.sub([tick_interval])

    def tick(self, *args):
        ds = orderbook_dataset(split=False,
                               shuffle=False,
                               labeled=False,
                               **self._kwargs)

        model = ViTForImageClassification.from_pretrained(PATH)

        image = ds[-1]["pixel_values"]
        feature_extractor = ViTFeatureExtractor.from_pretrained(PATH)
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        alog.info(logits)

        predicted_class_idx = logits.argmax(-1).item()

        alog.info(predicted_class_idx)

        self.publish(self.prediction_channel, predicted_class_idx)

        meas = vars(Measurement(measurement=f'prediction_{self.symbol}',
                    time=DateTimeUtils.now(), fields={
                        'prediction': predicted_class_idx
                    }))

        self.write_points([meas], time_precision='s')




@click.command()
@click.option("--additional-group-by", "-G", default="10Min", type=str)
@click.option("--cache", is_flag=True)
@click.option("--database_name", "-d", default="binance", type=str)
@click.option("--frame-width", "-F", default=224, type=int)
@click.option("--depth", default=72, type=int)
@click.option("--group-by", "-g", default="30s", type=str)
@click.option("--tick-interval", "-t", default="30s", type=str)
@click.option("--interval", "-i", default="10m", type=str)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--offset-interval", "-o", default="3h", type=str)
@click.option("--plot", "-p", is_flag=True)
@click.option("--round-decimals", "-D", default=4, type=int)
@click.option("--sequence-length", "-l", default=48, type=int)
@click.option("--tick", is_flag=True)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    PredictionEmitter(**kwargs)


if __name__ == "__main__":
    main()
