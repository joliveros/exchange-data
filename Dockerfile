FROM registry.rubercubic.com:5001/exchange-data:base

ENV NAME exchange-data
ENV PATH=/home/${USER}/.local/bin:$PATH

WORKDIR /home/joliveros/src

COPY . .

USER joliveros

RUN pip install -r requirements.txt

RUN pip install -e .

CMD ["./exchange-data"]
