FROM codequants.com:5000/exchange-data:base

ENV NAME exchange-data

WORKDIR /src

COPY . .

RUN bash -c "source ~/.bashrc && pip install -e ."

CMD ["./exchange-data"]
