FROM python:3.6.4

ENV NAME exchange-data

COPY . /src

WORKDIR /src

RUN pip install --upgrade pip

RUN pip install -r requirements.txt -r requirements-test.txt

CMD ["./exchange-data"]
