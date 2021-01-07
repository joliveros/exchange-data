FROM registry.rubercubic.com:5001/exchange-data:base

ENV NAME exchange-data

WORKDIR /home/joliveros/src

COPY . .

USER joliveros

RUN pip install -r requirements.txt

RUN pip install -e .

CMD ["./exchange-data"]
