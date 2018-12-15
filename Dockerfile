FROM continuumio/miniconda

ENV NAME exchange-data

COPY . /src

WORKDIR /src

RUN apt-get update && apt-get install -y build-essential

RUN bash -c "conda env create -f environment.yml"

CMD ["./exchange-data"]
