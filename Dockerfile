FROM python:3.6.1-slim

ENV NAME save-bitmex-data

ADD . /src

WORKDIR /src

RUN python ./setup.py install

CMD BATCH_SIZE=1 save-bitmex-data "XBTUSD"
