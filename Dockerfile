FROM registry.rubercubic.com:5001/exchange-data:base

ENV NAME exchange-data

USER root

WORKDIR /src

COPY . .

RUN pip install -e .

CMD ["./exchange-data"]
