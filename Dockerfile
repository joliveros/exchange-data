FROM python:3.6.2-slim

ENV NAME exchange-data

COPY . /src

WORKDIR /src

RUN python ./setup.py install

CMD ["exchange-data"]
