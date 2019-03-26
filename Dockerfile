FROM codequants.com:5000/exchange-data:base

ENV NAME exchange-data
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/compat/:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /src

COPY . .

#RUN bash -c "conda env update -f environment.yml"

RUN bash -c "source ~/.bashrc && pip install -e ."

CMD ["./exchange-data"]
