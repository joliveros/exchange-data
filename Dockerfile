FROM python:3.6.2-slim

ENV NAME save-bitmex-data

COPY . /src

WORKDIR /src

RUN python ./setup.py install

CMD ["save-bitmex-data", "XBTUSD"]
